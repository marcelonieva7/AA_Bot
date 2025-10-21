from phoenix.otel import register

def _init_tracer():
    project_name = "AA_CHATBOT"
    tp = register(protocol="http/protobuf", project_name=project_name)
    return tp.get_tracer("chatbot-ta")

tracer = _init_tracer()
