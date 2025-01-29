



   import streamlit as st
   import torch
   from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

   # Load the Parler-TTS model and processor
   @st.cache_resource
   def load_model():
       processor = AutoProcessor.from_pretrained("parler-tts/parler-tts-mini")
       model = AutoModelForSpeechSeq2Seq.from_pretrained("parler-tts/parler-tts-mini")
       return processor, model

   processor, model = load_model()

   def generate_audio(text):
       inputs = processor(text, return_tensors="pt")
       with torch.no_grad():
           speech = model.generate_speech(inputs["input_ids"])
       return speech.numpy()

   def main():
       st.title("🎤 Text-to-Audio Generator with Parler-TTS")
       st.write("Enter text below and convert it to high-quality audio.")

       # Text input
       user_text = st.text_area("Enter your text:", placeholder="Type here...", height=200)

       if st.button("🎧 Generate Audio"):
           if user_text.strip():
               with st.spinner("Generating audio..."):
                   audio_data = generate_audio(user_text)
                   st.audio(audio_data, format="audio/wav")
                   st.download_button(
                       label="📥 Download Audio",
                       data=audio_data,
                       file_name="generated_audio.wav",
                       mime="audio/wav",
                   )
           else:
               st.error("Please enter some text before generating audio!")

   if __name__ == "__main__":
       st.set_page_config(
           page_title="Text-to-Audio Generator with Parler-TTS",
           layout="wide",
           initial_sidebar_state="collapsed",
       )
       main()
   

