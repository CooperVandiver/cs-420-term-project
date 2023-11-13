import torch
from peft import PeftModel, PeftConfig
from traceback import print_exc
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

model_name = 'mosaicml/mpt-7b-chat'
adapters_name = './data/'

print('--- Loading model: mosaicml/mpt-7b-chat ---')
m = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
#    device_map={"":0}
)

print('--- Adding adapter: physics assistant ---')
peft_config = PeftConfig.from_pretrained('./data/physics')
m.add_adapter(peft_config, adapter_name='physics')

print('--- Adding adapter: biology assistant ---')
# TODO

m.enable_adapters()

# Setting default adapter
m.set_adapter('physics')

#m = PeftModel.from_pretrained(m, adapters_name)
#m = m.merge_and_unload()
tok = AutoTokenizer.from_pretrained(model_name)

while True:
    prompt = input('> ') + '\nA.'

    inputs = tok(prompt, return_tensors='pt').input_ids

    try:
        with torch.no_grad():
            outputs = m.generate(
                input_ids = inputs,
                do_sample=True,
                max_new_tokens=40
            )

            outputs = tok.batch_decode(
                outputs.detach().cpu().numpy()
#outputs.detach().cpu().numpy(),
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

            print(
                outputs
            )
    except Exception as e:
        print_exc()

