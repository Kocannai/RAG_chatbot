import gradio as gr
import httpx
import json

FASTAPI_URL = "http://127.0.0.1:8000"
#MODEL_PATH = "Qwen/Qwen2-1.5B-Instruct"
MODEL_PATH = "AITeamVN/Vi-Qwen2-1.5B-RAG"

def upload_pdf_to_backend(file):
    """Gửi file PDF lên endpoint /add_law_pdf."""
    if file is None:
        return "Vui lòng chọn một file PDF."
    files = {'file': (file.name, open(file.name, 'rb'), 'application/pdf')}
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(f"{FASTAPI_URL}/add_law_pdf", files=files)
        if response.status_code == 200:
            return f"Tải lên thành công! {response.json().get('message', '')}"
        else:
            return f" Lỗi: {response.text}"
    except httpx.RequestError as e:
        return f" Lỗi kết nối đến backend: {e}"

def reset_backend():
    """Gửi yêu cầu reset hệ thống."""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{FASTAPI_URL}/reset")
            return response.json().get('message', 'Không có phản hồi.')
    except httpx.RequestError as e:
        return f" Lỗi kết nối: {e}"

async def chatbot_response(message: str, history: list, context_state: dict):
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, context_state

    payload = {
        "model_path": MODEL_PATH,
        "question": message,
        "context_state": context_state
    }

    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"{FASTAPI_URL}/chat", json=payload) as response:
                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type:
                    response_bytes = await response.aread()
                    data = json.loads(response_bytes)
                    
                    context_text = data.get('introductory_question', "Vui lòng trả lời:") + "\n\n"
                    for q in data.get('context_questions', []):
                        context_text += f"- {q['question']}\n"
                    context_text += "\n*(Trả lời mỗi câu hỏi trên một dòng riêng biệt rồi nhấn Gửi.)*"
                    
                    history[-1]["content"] = context_text
                    
                    new_context_state = {
                        "original_question": data.get("original_question"),
                        "context_questions": data.get("context_questions")
                    }
                    yield history, new_context_state
                    return

                elif "text/event-stream" in content_type:
                    async for chunk in response.aiter_text():
                        if chunk:
                            full_response += chunk
                            history[-1]["content"] = full_response
                            yield history, {}
                    yield history, {}
                    return
                
                else:
                    error_text = await response.aread()
                    history[-1]["content"] = f" Lỗi không xác định: {error_text.decode()}"
                    yield history, {}

    except httpx.RequestError as e:
        history[-1]["content"] = f" Lỗi kết nối: {e}"
        yield history, {}

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Chatbot Pháp Luật VN") as demo:
    context_state = gr.State({})

    gr.Markdown("Chatbot Tư vấn Pháp luật Việt Nam")
    gr.Markdown("Tải lên file văn bản luật (PDF), sau đó bắt đầu hỏi đáp.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="Tải lên file luật (PDF)", file_types=[".pdf"])
            upload_button = gr.Button("Xử lý File", variant="primary")
            upload_status = gr.Textbox(label="Trạng thái xử lý file", interactive=False)
            
            reset_button = gr.Button("Reset Hệ thống")
            reset_status = gr.Textbox(label="Trạng thái Reset", interactive=False)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Hỏi đáp",
                height=600,
                type='messages',
                avatar_images=(None, "https://tinhdoan.caobang.gov.vn/uploads/news/2024_10/logo-cuoc-thi-tim-hieu-phap-luat.png")
            )
            msg_textbox = gr.Textbox(
                label="Nhập câu hỏi của bạn ở đây...",
                placeholder="Ví dụ: quy định về hợp đồng là gì?",
                scale=7
            )
            clear_button = gr.ClearButton([msg_textbox, chatbot], value="Xóa cuộc trò chuyện")

    upload_button.click(upload_pdf_to_backend, inputs=[pdf_file], outputs=[upload_status])
    reset_button.click(reset_backend, outputs=[reset_status])
    
    msg_textbox.submit(
        chatbot_response,
        inputs=[msg_textbox, chatbot, context_state],
        outputs=[chatbot, context_state]
    )

demo.launch(share=True)