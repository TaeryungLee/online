import sys
import subprocess
from pathlib import Path

def gif_to_mp4(input_gif: str, output_mp4: str):
    in_path = Path(input_gif)
    out_path = Path(output_mp4)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input GIF not found: {in_path}")

    # 출력 디렉토리가 없으면 생성
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg 명령어 구성
    # -y : 덮어쓰기
    # -movflags +faststart : 웹 재생 최적화
    # -pix_fmt yuv420p : 호환성 높은 픽셀 포맷
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_path),
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

def main():
    if len(sys.argv) != 3:
        print("Usage: python gif2mp4.py <input.gif> <output.mp4>")
        sys.exit(1)

    input_gif = sys.argv[1]
    output_mp4 = sys.argv[2]

    try:
        gif_to_mp4(input_gif, output_mp4)
        print(f"Saved: {output_mp4}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()