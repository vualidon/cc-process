import concurrent.futures
import os
import json
import time
import requests
import gzip
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import trafilatura
import fasttext
from huggingface_hub import hf_hub_download


# model for fasttext
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)
MAX_WORKERS = 6

def download_and_extract(file):
    try:
        # download the data
        prefix = 'https://data.commoncrawl.org/'
        url = prefix + file
        file_name = url.split('/')[-1]
        chunk_size = 1024 * 1024 * 32

        if not os.path.exists('cc-main-2024-26/' + file_name):
            print(f'Downloading {file_name}...')
            # download and save to a file
            response = requests.get(url)
            response.raise_for_status()

            with open('cc-main-2024-26/' + file_name, 'wb') as f:
                f.write(response.content)

            print(f'Downloaded {file_name} successfully!')
        else:
            print(f'File {file_name} already exists!')

        print(f'Extracting {file_name}...')
        # unzip and save to a file
        with gzip.open('cc-main-2024-26/' + file_name, 'rb') as f:
            with open('cc-main-2024-26/' + file_name[:-3], 'wb') as f_out:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
        print(f'Extracted file {file_name}')
        os.remove('cc-main-2024-26/' + file_name)
    except Exception as e:
        print('ERROR in download_and_extract:', e)

def classify_language(text):
    return model.predict(text)

def process_record(warc_file, file_to_save):
    with open(warc_file, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                url = record.rec_headers.get_header('WARC-Target-URI')
                html_content = record.content_stream().read().decode('utf-8', 'ignore')
                if not html_content:
                    continue
                
                content = trafilatura.extract(html_content)
                if not content:
                    continue

                lang = classify_language(content.replace('\n', ' '))[0][0]
                if lang == '__label__vie_Latn':
                    with open(file_to_save, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"url": url, "content": content}, ensure_ascii=False) + '\n')

def process_warc(warc_url, file_to_save):
    start_time = time.time()
    local_filename = 'cc-main-2024-26/' + warc_url.split('/')[-1][:-3]
    print(f'Processing {warc_url}...')
    try:
        # Download WARC file
        download_and_extract(warc_url)
        
        # Extract HTML content
        process_record(local_filename, file_to_save)
        
        # Remove processed WARC file
        os.remove(local_filename)

        return f"Processed {warc_url}"
    except Exception as e:
        return f"Error processing {warc_url}: {str(e)}"
    finally:
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

def main():
    with open('warc.paths', 'r') as f:
        warc_urls = f.readlines()

    warc_urls = [warc_url.strip() for warc_url in warc_urls]
    # warc_urls = warc_urls[24:36]

    for i in range(24, 36, MAX_WORKERS):
        file_to_save = f'cc-main-2024-26-vi/CC-MAIN-2024-26-VI-000{i // 100}.jsonl'
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for warc_url in warc_urls[i:i+MAX_WORKERS]:
                future = executor.submit(process_warc, warc_url, file_to_save)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                print(future.result())

if __name__ == "__main__":
    main()