# QLoRA imports
import torch
from peft import (
    PeftModel,
    PeftConfig
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

# Web imports
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for
)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

experts = ['Default', 'Physics', 'Biology']

# Loading model and adapters
model_id = 'facebook/opt-1.3b'

m = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Physics adapter
peft_config = PeftConfig.from_pretrained('./data/physics')
m.add_adapter(peft_config, adapter_name='Physics')

# Biology adapter
peft_config = PeftConfig.from_pretrained('./data/biology')
m.add_adapter(peft_config, adapter_name='Biology')

tok = AutoTokenizer.from_pretrained(model_id)

@app.route('/')
def index():
    return redirect(url_for('chat', expert='Default'))

@app.post('/expert')
@app.post('/expert/')
def selectExpert():
    body = request.form
    expert_name = body.get('expert')
    if expert_name == None or not expert_name in experts:
        return redirect(url_for('index'))
    return redirect(url_for('chat', expert=expert_name))

@app.route('/chat/<expert>')
def chat(expert):
    return render_template('index.html', experts=experts, expert=expert)

@app.post('/message')
@app.post('/message/')
def genMessage():
    body = request.get_json()
    expert = body.get('expert')
    if not expert:
        m.disable_adapters()
    else:
        if expert != 'Default' and expert in experts:
            print('--- Valid expert! ---')
            m.enable_adapters()
            m.set_adapter(expert)
        else:
            m.disable_adapters()

    prompt = body.get('message')
    if not prompt:
        return {'error': 'Message field is required'}

    print(f'--- Running {expert} ---')

    response_msg = ''

    prompt = prompt + '\nA.'

    inputs = tok(prompt, return_tensors='pt').input_ids

    try:
        print('--- Generating output ---')
        with torch.no_grad():
            outputs = m.generate(
                input_ids=inputs,
                do_sample=True,
                max_new_tokens=40
            )

            outputs = tok.batch_decode(
                outputs.detach().cpu().numpy()
            )
            outputs = outputs[0]

            outputs = outputs.split('\nA.')
            outputs = outputs[1]

            if outputs.find('.') > 0:
                outputs = outputs.split('.')[:-1]
            
            if len(outputs[-1].strip()) <= 1:
                outputs = outputs[:-1]

            outputs = list(map(lambda a: a.strip(), outputs))

            outputs = '. '.join(outputs)

            if not outputs[-1] in '!?.':
                outputs += '.'

            try:
                outputs = outputs.split('\n')[0]
            except:
                pass

            response_msg = outputs
    except:
        pass

    return {'message': response_msg}
