import whisper

model = whisper.load_model("base") 

result = model.transcribe("C:\Temp\Chris\A Plan To Not Get Deported ¦ James Smith (128kbit_AAC).m4a", language="en", fp16=False, verbose=False) 
print(result["text"])