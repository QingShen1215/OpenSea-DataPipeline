import os
import time
import requests
from urllib.parse import urlparse

API_KEY = "c532c24129c540cb9259cfa23a043ee8"  # ← 把你的 key 填这里
HEADERS = {
    "X-API-KEY": API_KEY,
    "Accept": "application/json",
}

# BAYC 合约 + 链
CHAIN = "ethereum"
CONTRACT = "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d"

BASE_URL = "https://api.opensea.io/api/v2"
SAVE_DIR = "bayc_images"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_file_ext_from_url(url: str) -> str:
    """
    根据 URL 猜一个文件后缀，默认 png
    """
    path = urlparse(url).path
    if "." in path:
        ext = path.split(".")[-1].lower()
        # 简单限制一下常见格式
        if ext in {"png", "jpg", "jpeg", "gif", "webp"}:
            return ext
    return "png"


def download_image(img_url: str, save_path: str) -> bool:
    try:
        resp = requests.get(img_url, timeout=20)
        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
        else:
            print(f"[Download fail] {img_url} | status={resp.status_code}")
            return False
    except Exception as e:
        print(f"[Download error] {img_url} | {e}")
        return False


def main():
    print("🚀 Start downloading BAYC images via contract endpoint...")

    url = f"{BASE_URL}/chain/{CHAIN}/contract/{CONTRACT}/nfts"
    params = {"limit": 200}  # OpenSea 文档里 1–200 之间

    next_cursor = None
    total_downloaded = 0

    while True:
        if next_cursor:
            params["next"] = next_cursor

        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code != 200:
            print(f"[Error] status={resp.status_code} response={resp.text[:200]}")
            break

        data = resp.json()
        nfts = data.get("nfts", [])
        if not nfts:
            print("没有更多 nfts 了，结束。")
            break

        for nft in nfts:
            identifier = nft.get("identifier")  # token_id，字符串
            if identifier is None:
                continue

            # 有些字段在 metadata 里，有些在顶层
            metadata = nft.get("metadata") or {}
            img_url = (
                metadata.get("image_original_url")
                or metadata.get("image_url")
                or metadata.get("image")
                or nft.get("image_url")
            )

            if not img_url:
                print(f"[No image_url] token {identifier}")
                continue

            ext = get_file_ext_from_url(img_url)
            save_path = os.path.join(SAVE_DIR, f"{identifier}.{ext}")

            # 已经下载过就跳过
            if os.path.exists(save_path):
                print(f"[Skip] {identifier} already exists.")
                continue

            ok = download_image(img_url, save_path)
            if ok:
                total_downloaded += 1
                print(f"[OK] token {identifier} -> {save_path}")
            else:
                print(f"[Failed] token {identifier}")

            # 稍微慢一点，避免被限流
            time.sleep(0.2)

        # 翻页
        next_cursor = data.get("next")
        if not next_cursor:
            print("没有 next cursor 了，全部处理完毕。")
            break

        print(f"➡️ 继续下一页，next={next_cursor}")
        time.sleep(0.5)

    print(f"🎉 Done. Total downloaded: {total_downloaded}")


if __name__ == "__main__":
    main()
