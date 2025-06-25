import ai_edge_torch
 
def convert_to_tflite(model, input):
    edge_model = ai_edge_torch.convert(model.eval(), input)
 
    return edge_model
 
 
