from models.CGR import CGR
def get_model(model_name, args):
    name = model_name.lower()


    if name == "cgr":
        return CGR(args)
    else:
        assert 0
