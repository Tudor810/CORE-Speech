"""Background services that the views drive.

Every service in here is *self-contained* (camera, sign engine, TTS,
voice manager, training, audio recorder) so the views never have to
touch threading or asyncio directly — they call sync methods and
register callbacks.
"""
