from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
  secret_key="sk-lf-a96d608f-1edd-4b30-bf76-3db501c6e9ff",
  public_key="pk-lf-94bc2370-6aec-4671-b8fe-3ddcc099d5d1",
  host="https://langfuse.formularizer.com",
  tags=["vividbot"],
)
