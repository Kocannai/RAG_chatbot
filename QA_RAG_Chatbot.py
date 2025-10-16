# uvicorn QA_RAG_Chatbot:app --host 127.0.0.1 --port 8000 --reload

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import asyncio
from sentence_transformers import SentenceTransformer
import faiss
import fitz
import numpy as np
import re
import logging
import gc
import json
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG_Chatbot_v2")

logger.info("Đang tải mô hình embedding...")
embedder = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
dimension = embedder.get_sentence_embedding_dimension()
logger.info("Tải mô hình embedding thành công.")

model = None
tokenizer = None
current_model_path = ""

documents_store = []
all_article_embeddings = np.array([], dtype=np.float32).reshape(0, dimension)
article_index = faiss.IndexFlatL2(dimension)
law_index = faiss.IndexFlatL2(dimension)
law_names = []

bm25_corpus = []
bm25_index = None

def preprocess_law_text(text: str):
    processed_docs = []
    chunks = re.split(r'(?=\n(?:BỘ LUẬT|LUẬT)\s|\nĐiều \d+)', text)
    current_law_name = "Chưa xác định"
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        law_name_match = re.search(r'^(BỘ LUẬT|LUẬT)\s+([^\n]+)', chunk, re.UNICODE)
        if law_name_match:
            law_type = law_name_match.group(1).strip()
            law_title = law_name_match.group(2).strip().replace('\n', ' ').title()
            current_law_name = f"Bộ luật {law_title}" if law_type == "BỘ LUẬT" else f"Luật {law_title}"
            logger.info(f"Đã xác định bộ luật mới: {current_law_name}")
            continue
        if chunk.startswith("Điều"):
            article_match = re.search(r'^(Điều \d+)\.?(.*)', chunk, re.DOTALL | re.UNICODE)
            if article_match:
                article_number = article_match.group(1).strip()
                content = article_match.group(2).strip().replace('\n', ' ').strip()
                if len(content) > 20:
                    processed_docs.append({
                        "law_name": current_law_name,
                        "article_number": article_number,
                        "text": f"{article_number}: {content}"
                    })
    logger.info(f"Đã xử lý và tách được {len(processed_docs)} điều luật.")
    return processed_docs

async def analyze_query_for_context(query: str, llm_model, llm_tokenizer) -> dict:
    logger.info(f"Phân tích bối cảnh (CoT) cho câu hỏi: '{query}'")
    context_analysis_prompt = """Bạn là một trợ lý pháp lý bậc thầy. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và xác định các thông tin bối cảnh CẦN THIẾT để có thể đưa ra câu trả lời chính xác nhất. Hãy suy nghĩ từng bước.

**Quy trình suy luận:**

1.  **<suy_nghĩ>**
    - **Chủ đề pháp lý chính:** Xác định lĩnh vực pháp lý cốt lõi của câu hỏi (ví dụ: Tội trốn thuế, Tranh chấp hợp đồng, Thủ tục ly hôn).
    - **Các yếu tố ảnh hưởng:** Liệt kê các yếu tố chính trong luật pháp có thể làm thay đổi câu trả lời cho chủ đề này. Ví dụ, đối với "Tội trốn thuế", các yếu tố ảnh hưởng là: (1) Đối tượng (cá nhân/pháp nhân), (2) Số tiền, (3) Lịch sử vi phạm.
    - **Thông tin đã có:** Kiểm tra xem câu hỏi của người dùng đã cung cấp thông tin cho các yếu tố nào chưa.
    - **Thông tin còn thiếu:** Liệt kê những yếu tố quan trọng còn thiếu.
    - **Kết luận:** Dựa trên thông tin còn thiếu, quyết định xem câu hỏi đã `CLEAR` hay cần `CONTEXT_NEEDED`.
    **</suy_nghĩ>**

2.  **<json_output>**
    - Nếu kết luận là `CLEAR`, chỉ trả về chuỗi: `"CLEAR"`.
    - Nếu kết luận là `CONTEXT_NEEDED`, hãy tạo một JSON chứa các câu hỏi để thu thập thông tin còn thiếu.
    **</json_output>**

---
**Ví dụ 1:**
- Câu hỏi của người dùng: "trốn thuế thì bị phạt thế nào?"
- Phân tích của bạn:
<suy_nghĩ>
- Chủ đề pháp lý chính: Tội trốn thuế.
- Các yếu tố ảnh hưởng: Đối tượng (cá nhân/pháp nhân), số tiền trốn thuế, lịch sử vi phạm (lần đầu hay tái phạm).
- Thông tin đã có: Không có.
- Thông tin còn thiếu: Cả ba yếu tố trên đều thiếu.
- Kết luận: CONTEXT_NEEDED.
</suy_nghĩ>
<json_output>
{{
  "context_needed": true,
  "introductory_question": "Để xác định chính xác mức phạt cho tội trốn thuế, tôi cần bạn cung cấp thêm một vài thông tin về trường hợp của mình:",
  "context_questions": [
    {{"id": "entity_type", "question": "Hành vi này được thực hiện bởi cá nhân hay một doanh nghiệp (pháp nhân thương mại)?"}},
    {{"id": "amount", "question": "Số tiền trốn thuế là khoảng bao nhiêu?"}},
    {{"id": "history", "question": "Đây có phải là lần vi phạm đầu tiên không?"}}
  ]
}}
</json_output>

---
**Ví dụ 2:**
- Câu hỏi của người dùng: "bạn tôi là cá nhân, trốn thuế 500 triệu, lần đầu vi phạm thì bị phạt tù không?"
- Phân tích của bạn:
<suy_nghĩ>
- Chủ đề pháp lý chính: Tội trốn thuế.
- Các yếu tố ảnh hưởng: Đối tượng, số tiền, lịch sử vi phạm.
- Thông tin đã có: Đối tượng (cá nhân), số tiền (500 triệu), lịch sử vi phạm (lần đầu).
- Thông tin còn thiếu: Không có yếu tố quan trọng nào bị thiếu.
- Kết luận: CLEAR.
</suy_nghĩ>
<json_output>
"CLEAR"
</json_output>

---
Bây giờ, hãy phân tích câu hỏi sau.
**Câu hỏi của người dùng:** "{question}"
**Phân tích của bạn:**
"""
    prompt = context_analysis_prompt.format(question=query)
    messages = [{"role": "system", "content": "Bạn là một trợ lý pháp lý chuyên phân tích yêu cầu."}, {"role": "user", "content": prompt}]
    chat_prompt = llm_tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    inputs = llm_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(llm_model.device)
    with torch.inference_mode():
        output_ids = llm_model.generate(**inputs, max_new_tokens=512, pad_token_id=llm_tokenizer.eos_token_id)
    response_text = llm_tokenizer.batch_decode(output_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0].strip()
    try:
        json_block_match = re.search(r'<json_output>(.*?)</json_output>', response_text, re.DOTALL)
        if json_block_match:
            json_text = json_block_match.group(1).strip()
            if "CLEAR" in json_text:
                 logger.info("Phân tích CoT kết luận câu hỏi đã rõ ràng.")
                 return {"context_needed": False}
            context_data = json.loads(json_text)
            logger.info("Phân tích CoT kết luận cần thêm bối cảnh.")
            return context_data
        else:
             logger.warning("Không tìm thấy tag <json_output> trong phản hồi của LLM.")
             return {"context_needed": False}
    except Exception as e:
        logger.error(f"Lỗi khi parse JSON CoT từ LLM: {e}. Coi như câu hỏi đã rõ ràng.")
        return {"context_needed": False}

async def inference_stream(llm_model, llm_tokenizer, input_prompt: str, system_prompt: str):
    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}]
    chat_prompt = llm_tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    inputs = llm_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(llm_model.device)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2048, pad_token_id=llm_tokenizer.eos_token_id)
    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text
        await asyncio.sleep(0.01)

def add_vietnamese_diacritics(query: str, llm_model, llm_tokenizer) -> str:
    if len(query.split()) < 2 or any(c in 'ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ' for c in query.lower()):
        return query

    logger.info(f"Phát hiện câu không dấu, đang thêm dấu cho: '{query}'")
    diacritics_prompt = """Bạn là một chuyên gia ngôn ngữ Tiếng Việt. Nhiệm vụ của bạn là đọc một câu không dấu và thêm các dấu câu (sắc, huyền, hỏi, ngã, nặng) và dấu mũ (â, ê, ô) một cách chính xác nhất để tạo thành một câu hoàn chỉnh có nghĩa. Chỉ trả về câu đã được thêm dấu, không thêm bất kỳ lời giải thích nào.

Ví dụ:
- Input: "toi muon hoi ve luat dat dai" -> Output: "tôi muốn hỏi về luật đất đai"
- Input: "tron thue bi phat bao nhieu nam tu" -> Output: "trốn thuế bị phạt bao nhiêu năm tù"

Bây giờ, hãy thêm dấu cho câu sau:
Câu không dấu: "{question}"
Câu đã thêm dấu:"""
    prompt = diacritics_prompt.format(question=query)
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = llm_tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    inputs = llm_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(llm_model.device)
    with torch.inference_mode():
        output_ids = llm_model.generate(**inputs, max_new_tokens=int(len(query) * 2), pad_token_id=llm_tokenizer.eos_token_id)
    accented_query = llm_tokenizer.batch_decode(output_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0].strip()
    return accented_query if accented_query else query

def rewrite_query_if_needed(original_query: str, llm_model, llm_tokenizer) -> str:
    logger.info(f"Đang kiểm tra và có thể viết lại câu hỏi: '{original_query}'")
    rewrite_prompt_template = """Bạn là một chuyên gia ngôn ngữ và pháp luật Việt Nam. Nhiệm vụ của bạn là đọc câu hỏi của người dùng và viết lại nó một cách rõ ràng, đầy đủ ngữ pháp hơn.
- **QUAN TRỌNG NHẤT: Phải giữ nguyên ý định và ngữ nghĩa gốc của người dùng.** Đặc biệt cẩn thận với các từ đồng âm hoặc gần âm nhưng khác nghĩa (ví dụ: "thuê" và "thuế").
- Sửa lỗi chính tả, ngữ pháp.
- Viết đầy đủ các từ viết tắt phổ biến (ví dụ: "blhs" -> "bộ luật hình sự", "csgt" -> "cảnh sát giao thông").
- Nếu câu hỏi đã rõ ràng, hãy trả về CHÍNH NÓ.
- Chỉ trả về câu hỏi đã được viết lại, không thêm lời giải thích.

Ví dụ:
- Input: "hợp đồng bj vô hiệu là ntn?" -> Output: Hợp đồng bị vô hiệu là như thế nào?
- Input: "tội trốn thuế bị pnhu a" -> Output: Tội trốn thuế bị phạt như thế nào?

Bây giờ, hãy viết lại câu hỏi sau:
Câu hỏi gốc: "{question}"
Câu hỏi đã viết lại:"""
    prompt = rewrite_prompt_template.format(question=original_query)
    messages = [{"role": "system", "content": "Bạn là một trợ lý ngôn ngữ thông minh."}, {"role": "user", "content": prompt}]
    chat_prompt = llm_tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    inputs = llm_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(llm_model.device)
    with torch.inference_mode():
        output_ids = llm_model.generate(**inputs, max_new_tokens=100, pad_token_id=llm_tokenizer.eos_token_id)
    rewritten_query = llm_tokenizer.batch_decode(output_ids[:, inputs['input_ids'][0].shape[-1]:], skip_special_tokens=True)[0].strip()
    return rewritten_query if rewritten_query else original_query


def route_query(question: str, llm_model, llm_tokenizer) -> str:
    logger.info(f"Đang định tuyến cho câu hỏi: '{question}'")
    routing_prompt_template = """Bạn là một chuyên gia phân loại câu hỏi pháp lý. Nhiệm vụ của bạn là phân loại câu hỏi của người dùng vào MỘT trong hai loại sau đây. Hãy suy nghĩ kỹ trước khi quyết định.

1. `direct_lookup`: Dành cho các câu hỏi yêu cầu tra cứu một điều, khoản, hoặc bộ luật CỤ THỂ. Câu hỏi PHẢI chứa số hiệu (ví dụ: 'Điều 123', 'Khoản 2 Điều 5') VÀ thường đi kèm tên một bộ luật.
   - Ví dụ chuẩn: "Điều 200 Bộ luật Hình sự nói về tội gì?", "Khoản 1 Điều 15 Luật Giao thông đường bộ", "xem giúp tôi điều 300 của bộ luật dân sự".

2. `vector_search`: Dành cho tất cả các câu hỏi còn lại. Đây là những câu hỏi hỏi về khái niệm, tình huống, định nghĩa, hoặc một tội danh chung chung mà KHÔNG chỉ rõ số điều.
   - Ví dụ chuẩn: "Thế nào là tội vu khống?", "Quy định về hợp đồng lao động?", "Làm lính đánh thuê thì bị phạt thế nào?", "Trốn thuế bị xử lý ra sao?".

Dựa vào phân tích trên, hãy phân loại câu hỏi sau. Chỉ trả về MỘT TỪ DUY NHẤT: `direct_lookup` hoặc `vector_search`.

Câu hỏi: "{question}"
Phân loại:"""
    prompt = routing_prompt_template.format(question=question)
    messages = [{"role": "system", "content": "Bạn là một trợ lý phân loại truy vấn thông minh và chính xác."}, {"role": "user", "content": prompt}]
    chat_prompt = llm_tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    inputs = llm_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(llm_model.device)
    with torch.inference_mode():
        output_ids = llm_model.generate(**inputs, max_new_tokens=5, pad_token_id=llm_tokenizer.eos_token_id)
    result = llm_tokenizer.batch_decode(output_ids[:, inputs['input_ids'][0].shape[-1]:], skip_special_tokens=True)[0].strip().lower()
    if "direct_lookup" in result:
        return "direct_lookup"
    return "vector_search"

class ChatIn(BaseModel):
    model_path: str
    question: str
    context_state: dict | None = None
    
@app.post("/add_law_pdf")
async def add_law_pdf(file: UploadFile = File(...)):
    global documents_store, all_article_embeddings, article_index, law_index, law_names, bm25_corpus, bm25_index
    try:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "\n".join(page.get_text("text") for page in doc)
        new_docs = preprocess_law_text(full_text)
        if not new_docs:
            return JSONResponse(status_code=400, content={"error": "Không tìm thấy điều luật nào trong file."})
        documents_store.extend(new_docs)
        article_texts = [d['text'] for d in new_docs]
        batch_size = 64
        for i in range(0, len(article_texts), batch_size):
            batch_texts = article_texts[i:i + batch_size]
            new_article_embeds = embedder.encode(batch_texts, convert_to_numpy=True).astype('float32')
            article_index.add(new_article_embeds)
            if all_article_embeddings.size == 0:
                all_article_embeddings = new_article_embeds
            else:
                all_article_embeddings = np.vstack([all_article_embeddings, new_article_embeds])
        
        tokenized_articles = [ViTokenizer.tokenize(re.sub(r'[^\w\s]', '', doc).lower()).split() for doc in article_texts]
        bm25_corpus.extend(tokenized_articles)
        bm25_index = BM25Okapi(bm25_corpus)
        
        unique_laws = {d['law_name'] for d in new_docs}
        for law_name in unique_laws:
            if law_name not in law_names:
                law_names.append(law_name)
                law_full_text = " ".join([d['text'] for d in documents_store if d['law_name'] == law_name])
                law_embed = embedder.encode([law_full_text], convert_to_numpy=True).astype('float32')
                law_index.add(law_embed)
        return {"message": f"Thêm thành công {len(new_docs)} điều luật. Hệ thống hiện có {len(documents_store)} điều."}
    except Exception as e:
        logger.error(f"Lỗi tại /add_law_pdf: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def rag_chat_stream(input_data: ChatIn):
    global model, tokenizer, current_model_path, documents_store, bm25_index
    try:
        if current_model_path != input_data.model_path or model is None:
            logger.info(f"Đang tải hoặc thay đổi mô hình LLM sang '{input_data.model_path}'...")
            del model, tokenizer
            gc.collect(); torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(input_data.model_path)
            model = AutoModelForCausalLM.from_pretrained(input_data.model_path, device_map="auto", dtype=torch.bfloat16)
            current_model_path = input_data.model_path
            logger.info("Tải mô hình LLM thành công!")

        final_question_for_rag = ""
        user_prompt = ""
        system_prompt = ""

        if input_data.context_state and 'original_question' in input_data.context_state:
            state = input_data.context_state
            original_question = state['original_question']
            context_questions = state['context_questions']
            user_answers_text = input_data.question
            
            retrieval_query = f"{original_question} với các chi tiết sau: {user_answers_text.replace(chr(10), ', ')}"
            final_question_for_rag = retrieval_query
            route_decision = "vector_search"
            logger.info(f"Đã tổng hợp bối cảnh. Truy vấn tìm kiếm: '{final_question_for_rag}'")
            
            system_prompt = "Bạn là một chuyên gia pháp lý cực kỳ cẩn thận và logic. Hãy suy nghĩ từng bước để đưa ra kết luận chính xác."
            
            synthesis_prompt_with_cot = """**Nhiệm vụ:** Dựa vào các điều luật trong 'Ngữ cảnh', hãy phân tích 'Tình huống' của người dùng và đưa ra kết luận pháp lý cuối cùng.

**Quy trình suy luận bắt buộc:**

1.  **<suy_nghĩ>**
    - **Sàng lọc Ngữ cảnh:** Đọc qua tất cả các điều luật được cung cấp. Xác định điều luật nào là **liên quan trực tiếp** đến chủ đề chính trong tình huống của người dùng. Loại bỏ hoàn toàn các điều luật không liên quan.
    - **Trích xuất Dữ kiện:** Liệt kê các dữ kiện quan trọng từ 'Tình huống' của người dùng (ví dụ: đối tượng là ai, số tiền bao nhiêu, có tình tiết tăng nặng/giảm nhẹ không).
    - **Áp dụng Luật:** Đối chiếu từng dữ kiện với các khoản, điểm trong điều luật liên quan đã được sàng lọc. Xác định chính xác khoản, điểm nào được áp dụng.
    - **Dự thảo Kết luận:** Dựa trên khoản, điểm đã xác định, viết ra mức phạt hoặc kết luận pháp lý.
    **</suy_nghĩ>**

2.  **<câu_trả_lời_cuối_cùng>**
    - Trình bày kết luận cuối cùng cho người dùng một cách rõ ràng, trực tiếp và chỉ dựa trên kết quả của bước suy nghĩ. **Không** bao gồm các điều luật không liên quan đã bị loại bỏ.
    **</câu_trả_lời_cuối_cùng>**

---
**Ngữ cảnh (Các điều luật tìm được):**
{context}

---
**Tình huống của người dùng:**
{user_case}

---
**Phân tích của bạn:**
"""
            user_case_summary = f"- Chủ đề: {original_question}\n"
            answers = user_answers_text.splitlines()
            for i, question_obj in enumerate(context_questions):
                if i < len(answers): user_case_summary += f"- {question_obj['question'].split('?')[0]}: {answers[i]}\n"
        else:
            accented_question = add_vietnamese_diacritics(input_data.question, model, tokenizer)
            rewritten_question = rewrite_query_if_needed(accented_question, model, tokenizer)
            context_result = await analyze_query_for_context(rewritten_question, model, tokenizer)
            if context_result.get("context_needed"):
                context_result['original_question'] = rewritten_question
                return JSONResponse(content=context_result)
            final_question_for_rag = rewritten_question
            route_decision = route_query(final_question_for_rag, model, tokenizer)
            
            system_prompt = (
            "Bạn là một trợ lý pháp luật chuyên nghiệp, có khả năng tổng hợp và giải thích thông tin một cách rõ ràng. "
            "Hãy trả lời câu hỏi của người dùng dựa trên ngữ cảnh được cung cấp. "
            "Nhiệm vụ của bạn là diễn giải các điều luật một cách chi tiết, dễ hiểu, sắp xếp các ý một cách logic. "
            "Nếu ngữ cảnh không chứa thông tin liên quan, hãy trả lời rằng bạn không tìm thấy thông tin trong tài liệu."
            )
            default_prompt_template = (
                "Dựa vào các điều luật trong phần 'Ngữ cảnh' dưới đây, hãy viết một câu trả lời chi tiết, đầy đủ cho 'Câu hỏi' của người dùng. "
                "Hãy trả lời câu hỏi của người dùng một cách chính xác và chỉ sử dụng thông tin được cung cấp, không diễn giải thêm và trình bày câu trả lời một cách mạch lạc.\n\n"
                "--- Ngữ cảnh ---\n"
                "{context}\n\n"
                "--- Câu hỏi ---\n"
                "{question}\n\n"
                "--- Trả lời chi tiết ---\n"
            )

        retrieved_docs_as_dict = []

        if route_decision == "vector_search":
            q_embedding = embedder.encode([final_question_for_rag], convert_to_numpy=True).astype('float32')
            _, I_law = law_index.search(q_embedding, k=2)
            top_law_names = [law_names[i] for i in I_law[0]]
            relevant_indices_global = [i for i, doc in enumerate(documents_store) if doc["law_name"] in top_law_names]
            if relevant_indices_global:
                candidate_embeddings = all_article_embeddings[relevant_indices_global]
                sub_index_faiss = faiss.IndexFlatL2(dimension)
                sub_index_faiss.add(candidate_embeddings)
                distances, I_faiss_local = sub_index_faiss.search(q_embedding, k=min(10, len(relevant_indices_global)))
                sub_corpus_bm25 = [bm25_corpus[i] for i in relevant_indices_global]
                sub_index_bm25 = BM25Okapi(sub_corpus_bm25)
                tokenized_query = ViTokenizer.tokenize(re.sub(r'[^\w\s]', '', final_question_for_rag).lower()).split()
                bm25_scores = sub_index_bm25.get_scores(tokenized_query)
                I_bm25_local = np.argsort(bm25_scores)[::-1][:min(5, len(relevant_indices_global))]
                fused_scores = {}
                k_rrf = 60
                for rank, idx_local in enumerate(I_faiss_local[0]):
                    original_doc_idx = relevant_indices_global[idx_local]
                    if original_doc_idx not in fused_scores: fused_scores[original_doc_idx] = 0
                    fused_scores[original_doc_idx] += 1 / (k_rrf + rank + 1)
                for rank, idx_local in enumerate(I_bm25_local):
                    if bm25_scores[idx_local] > 0:
                        original_doc_idx = relevant_indices_global[idx_local]
                        if original_doc_idx not in fused_scores: fused_scores[original_doc_idx] = 0
                        fused_scores[original_doc_idx] += 1 / (k_rrf + rank + 1)
                sorted_fused_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
                final_doc_indices = [doc_id for doc_id, score in sorted_fused_results[:3]]
                retrieved_docs_as_dict = [documents_store[i] for i in final_doc_indices]

        context_str = "Không tìm thấy điều luật nào phù hợp trong tài liệu."

        if retrieved_docs_as_dict:
            context_texts = [f"Trích {doc['law_name']}, {doc['article_number']}:\n{doc['text']}" for doc in retrieved_docs_as_dict]
            context_str = "\n\n---\n\n".join(context_texts)

        if 'synthesis_prompt_with_cot' in locals():
            user_prompt = synthesis_prompt_with_cot.format(context=context_str, user_case=user_case_summary)
        else:
            user_prompt = default_prompt_template.format(context=context_str, question=final_question_for_rag)

        async def response_generator():
            context_header = "**Các điều luật được tham khảo:**\n"
            if retrieved_docs_as_dict:
                for doc in retrieved_docs_as_dict:
                    context_header += f"- **{doc['law_name']}**: {doc['article_number']}\n"
            else:
                context_header += "- Không tìm thấy điều luật liên quan.\n"
            context_header += "\n---\n\n"
            yield context_header

            full_response = ""
            async for token in inference_stream(model, tokenizer, user_prompt, system_prompt=system_prompt):
                full_response += token
            
            final_answer_match = re.search(r'<câu_trả_lời_cuối_cùng>(.*?)</câu_trả_lời_cuối_cùng>', full_response, re.DOTALL)
            if final_answer_match:
                yield final_answer_match.group(1).strip()
            else:
                yield full_response

        return StreamingResponse(response_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng tại /chat: {e}", exc_info=True)
        async def error_generator(exc: Exception): yield f"Đã xảy ra lỗi nghiêm trọng: {exc}."
        return StreamingResponse(error_generator(e), media_type="text/event-stream")

@app.post("/reset")
def reset_system():
    global documents_store, all_article_embeddings, article_index, law_index, law_names, model, tokenizer, current_model_path, bm25_corpus, bm25_index
    documents_store, law_names, bm25_corpus = [], [], []
    all_article_embeddings = np.array([], dtype=np.float32).reshape(0, dimension)
    article_index, law_index, bm25_index = faiss.IndexFlatL2(dimension), faiss.IndexFlatL2(dimension), None
    del model, tokenizer
    model, tokenizer, current_model_path = None, None, ""
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Hệ thống đã được reset.")
    return {"message": "Đã reset và xóa toàn bộ dữ liệu."}

@app.get("/show_data")
def show_data():
    return {
        "current_llm_model": current_model_path,
        "total_laws": len(law_names),
        "total_articles": len(documents_store),
        "total_bm25_docs": len(bm25_corpus) if bm25_index else 0,
    }