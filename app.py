# Generate multi line comment with post body example
"""
{
    "model_name": "facebook/musicgen-small",
    "duration": 15,
    "prompt": "I love",
    "strategy": "loudness",
    "sampling": true,
    "top_k": 0,
    "top_p": 0.9,
    "temperature": 0.9
    "use_diffusion": true
    "use_custom": false
}
"""


from flask import Flask, request, send_file, abort, jsonify
from audiocraft.models import musicgen
from audiocraft.data.audio import audio_write
from audiocraft.models import MultiBandDiffusion
from dotenv import load_dotenv
import uuid
import io
import torch
import boto3
import os

app = Flask(__name__)

# load .env
load_dotenv()

AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

@app.route("/generate_music", methods=["POST"])
def generate_music():
    # Get the model name, duration, prompt, and strategy from the request body
    # print(request.json)
    # model_name = request.json.get("model_name")
    # duration = request.json.get("duration")
    # prompt = request.json.get("prompt")
    # strategy = request.json.get("strategy")
    # sampling = request.json.get("sampling")
    # top_k = request.json.get("top_k")
    # top_p = request.json.get("top_p")
    # temperature = request.json.get("temperature")
    # use_diffusion = request.json.get("use_diffusion")
    # use_custom = request.json.get("use_custom")
    keyword = request.json.get("keyword")

    model_name = "facebook/musicgen-large"
    duration = 30
    strategy = "loudness"
    sampling = True
    top_k = 0
    top_p = 0.9
    temperature = 0.9
    use_diffusion = False
    use_custom = False

    prompt = keyword

    # Check if the model name is valid
    if model_name not in [
        "facebook/musicgen-small",
        "facebook/musicgen-medium",
        "facebook/musicgen-large",
    ]:
        abort(
            400,
            "Invalid model name (facebook/musicgen-small, facebook/musicgen-medium, facebook/musicgen-large)",
        )

    # Check if the duration is valid
    if duration not in [15, 30, 60, 90, 120, 180]:
        abort(400, "Invalid duration in seconds (15, 30, 60, 90, 120, 180)")

    # Check if the prompt is valid
    if not isinstance(prompt, str):
        abort(400, "Invalid prompt (string)")

    # Check if the strategy is valid
    if strategy not in ["loudness", "peak", "clip"]:
        abort(400, "Invalid strategy (loudness, peak, clip)")

    # Check if the sampling is true or false
    if not isinstance(sampling, bool):
        abort(400, "Invalid sampling (true, false)")

    # Check if the top_k is valid
    if not isinstance(top_k, int):
        abort(400, "Invalid top_k (int)")

    # Check if the top_p is valid
    if not isinstance(top_p, float):
        abort(400, "Invalid top_p (float)")

    # Check if the temperature is valid
    if not isinstance(temperature, float):
        abort(400, "Invalid temperature (float)")

    # Check if the use_diffusion is valid
    if not isinstance(use_diffusion, bool):
        abort(400, "Invalid use_diffusion (true, false)")

    if not isinstance(use_custom, bool):
        abort(400, "Invalid use_custom (true, false)")

    # Print the request body
    print(request.json)

    # Generate a unique UUID for the generated .wav file
    myuuid = uuid.uuid4()

    # Load the specified model and set the generation parameters
    model = musicgen.MusicGen.get_pretrained(model_name, device="cuda")
    if use_custom:
        model.lm.load_state_dict(torch.load('models/lm_final.pt'))
    model.set_generation_params(
        duration=duration,
        use_sampling=sampling,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    # Generate the music using the specified prompt
    wav = model.generate([prompt], progress=True, return_tokens=True)

    # if use_diffusion:
    if use_diffusion:
        print("Using diffusion")
        mbd = MultiBandDiffusion.get_mbd_musicgen()
        diff = mbd.tokens_to_wav(wav[1])
        create_wav(diff, myuuid, model, strategy)
    else:
        print("Not using diffusion")
        create_wav(wav[0], myuuid, model, strategy)

    # Read the generated .wav file into memory
    with open(f"{str(myuuid)}.wav", "rb") as f:
        wav_data = f.read()

        s3_file_path = f"sound/{str(myuuid)}.wav"

        # S3에 오디오 파일 업로드
        print("S3 오디오 업로드 시작")
        s3 = s3_connection()
        s3.put_object(
            Bucket = BUCKET_NAME,
            Body = wav_data,
            Key = s3_file_path,
            ContentType = "audio/wav"
        )
        print("S3 오디오 업로드 종료")

    # JSON 형태로 응답 반환
    response = {
        "result": "success",
        "path": f"https://singsongchanson.s3.ap-northeast-2.amazonaws.com/{s3_file_path}"  # S3 버킷 URL
    }

    return jsonify(response)

    # Return the .wav file as a response
    # return send_file(
    #     io.BytesIO(wav_data),
    #     mimetype="audio/wav",
    #     as_attachment=True,
    #     download_name=f"{str(myuuid)}.wav",
    # )


def create_wav(output, myuuid, model, strategy):
    for idx, one_wav in enumerate(output):
        audio_write(
            f"{str(myuuid)}",
            one_wav.cpu(),
            model.sample_rate,
            strategy=strategy,
            loudness_compressor=True,
        )

def s3_connection() :
    s3 = boto3.client('s3', aws_access_key_id = AWS_ACCESS_KEY, aws_secret_access_key = AWS_SECRET_KEY)
    return s3

if __name__ == "__main__":
    app.run('0.0.0.0', port=7777, debug=True)