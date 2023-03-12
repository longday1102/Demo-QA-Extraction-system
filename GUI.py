from transformers import pipeline
from train import QA_model, tokenizer
import gradio as gr

pipe = pipeline('question-answering', model = QA_model, tokenizer = tokenizer)

def question_answering(Context, Question):
    res = pipe(question = Question, context = Context)
    answer = res['answer']
    drop_char = [',', ';', '/', '//']
    if answer[-1] in drop_char:
        answer = answer[:-1]
    return answer

app = gr.Interface(fn = question_answering,
                   inputs = [gr.Textbox(lines = 2, placeholder = 'Nhập tại đây...', label = 'Đoạn văn', max_lines = 2000),
                             gr.Textbox(lines = 2, placeholder = 'Nhập tại đây...', label = 'Câu hỏi', max_lines = 2000)],
                   outputs = gr.Textbox(label = 'Câu trả lời'),
                   title = 'BOT TRẢ LỜI CÂU HỎI TIẾNG VIỆT',
                   description = 'Hãy nhập đoạn văn và câu hỏi vào khoảng trống tương ứng, hệ thông sẽ đưa ra câu trả lời.',
                   theme = 'dark',
                   allow_flagging = 'never',
                   article = 'Hướng dẫn: Chọn submit để hiển thị câu trả lời hoặc Clear để xóa tất cả.')

app.launch(inline = False, share = True)
