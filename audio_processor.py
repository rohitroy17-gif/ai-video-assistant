import yt_dlp
from pydub import AudioSegment
import os
import shutil
import sys
import tempfile

RUNTIME_MARKER = "yt-dlp-ejs-node-v2"

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def _youtube_cookie_file() -> str | None:
    cookie_file = os.getenv("YOUTUBE_COOKIES_FILE")
    cookie_text = os.getenv("YOUTUBE_COOKIES")
    if not cookie_text:
        try:
            import streamlit as st
            cookie_text = st.secrets.get("YOUTUBE_COOKIES")
        except Exception:
            cookie_text = None

    if cookie_file:
        return cookie_file
    if not cookie_text:
        return None

    temporary_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8", delete=False
    )
    temporary_file.write(cookie_text)
    temporary_file.close()
    return temporary_file.name

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    node_path = shutil.which("node") or shutil.which("nodejs")
    if not node_path and os.path.isfile("/usr/bin/node"):
        node_path = "/usr/bin/node"
    if not node_path:
        raise RuntimeError(
            "Node.js is required for YouTube downloads. "
            "Add nodejs to packages.txt and redeploy the app."
        )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        "js_runtimes": {"node": {"path": node_path}},
        "remote_components": ["ejs:github"],
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    cookie_file = _youtube_cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    node_version = os.popen(f'"{node_path}" --version').read().strip()
    print(f"yt-dlp JavaScript runtime: {node_path} ({node_version})", flush=True)
    print(f"yt-dlp JavaScript runtime: {node_path} ({node_version})", file=sys.stderr, flush=True)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    finally:
        if cookie_file and not os.getenv("YOUTUBE_COOKIES_FILE"):
            try:
                os.unlink(cookie_file)
            except OSError:
                pass
    return filename


def convert_audio_to_wav(input_path:str)-> str:
    """convert any audio/video file to WAV format using pydub"""
    output_path= os.path.splitext(input_path)[0]+"_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio=audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path,chunk_minutes:int=10)->list:
    audio=AudioSegment.from_wav(wav_path)
    chunk_ms=chunk_minutes*60*1000
    chunks=[]

    for i,start in enumerate(range(0,len(audio),chunk_ms)):
        chunk=audio[start:start+chunk_ms]
        chunk_path=f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path,format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)

    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_audio_to_wav(source)

    chunks = chunk_audio(wav_path)

    print(f"Audio ready - {len(chunks)} chunk(s) created.")

    return chunks


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=HSuFmqN23YI"

    chunks = process_input(url)

    print("Created chunks:")
    for chunk in chunks:
        print(chunk)