server_url = "http://127.0.0.1:9991"
# server_url = "http://192.168.2.190:9991"


import queue
import random
import pillow_heif
import threading
import time
from io import BytesIO
import numpy as np
import cv2
import requests
from queue import Queue
import torch
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image

start_time = time.time()

half = True

pillow_heif.register_heif_opener()

work_queue = Queue()
uploader_queue = Queue()

maxwork = 5

main_should_run = True
uploader_should_run = True
last_job_time = 0


def upscaler_has_work():
    try:
        response = requests.get("/".join([server_url, "upscaling_get_pending_requests?models=diffuser"]), timeout=5)
        return int(response.text) > 0
    except Exception as e:
        print(e)
        return False


def canContinueToWork():
    return True


def fetch_job():
    try:
        response = requests.get("/".join(
            [server_url,
             "upscaling_fetch_work?models=diffuser&key=01874350981703051987035476132075964375601750416475810347563"]),
            timeout=5)
        if response.status_code == 204:
            return None  # No work
        if response.status_code != 200:
            print("Unexpected status:", response.status_code)
            return None

        result = response.json()
        data = requests.get(result["url"], timeout=10).content
        result["data"] = data
        return result
    except Exception as e:
        print("Error fetching job:", e)
        return None


def work_fetcher():
    global main_should_run, last_job_time

    while True:
        if work_queue.qsize() > maxwork or uploader_queue.qsize() > 10:
            time.sleep(1)
            continue

        t0 = time.time()

        job = fetch_job()
        if not job:
            ## check if other programs have work and exit if so
            if not canContinueToWork():
                main_should_run = False
                print("exit work_fetcher")
                return
            time.sleep(1)
            continue

        last_job_time = time.time()

        work_queue.put(job)
        print("work downloaded in ", time.time() - t0, "- requests waiting:", work_queue.qsize())


def upload_thread():
    while True:
        try:
            (client_id, original_filename, img_upscaled) = uploader_queue.get_nowait()
        except queue.Empty:
            if not uploader_should_run:
                print("exit upload_thread")
                return
            time.sleep(1)
            continue
        try:
            img_upscaled = np.array(img_upscaled)[:, :, ::-1]  # convert to BGR for OpenCV
            for t in range(5):
                print("start uploading ", t, " - ", client_id, " - ", original_filename)
                t0 = time.time()
                # Encode upscaled image to PNG
                success, encoded_image = cv2.imencode(".png", img_upscaled)
                if not success:
                    raise Exception("Failed to encode result image")

                new_filename = original_filename + ".png"

                files = {
                    "image": (new_filename, BytesIO(encoded_image.tobytes()), "image/png")
                }

                data = {
                    "client_id": client_id,
                    "original_filename": original_filename
                }

                r = requests.post("/".join([server_url, "upscaling_upload_result"]), data=data, files=files)
                if r.status_code != 200:
                    print("Upload failed:", r.status_code, r.text)
                else:
                    print("Upload successful:", original_filename, " - ", time.time() - t0, " - pending uploads:",
                          uploader_queue.qsize())
                    break

        except Exception as e:
            print("Upload error:", e)


def main():
    global main_should_run, uploader_should_run

    # Initialize all models at startup
    model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign="Q")
    model = model.half()
    model.ae_dtype = convert_dtype("bf16")
    model.model.dtype = convert_dtype("fp16")
    

    # Start worker threads
    work_fetcher_thread = threading.Thread(target=work_fetcher)
    work_fetcher_thread.start()

    uploader_thread1 = threading.Thread(target=upload_thread)
    uploader_thread1.start()

    print("Service ready! Processing jobs...")

    while True:
        try:
            job = work_queue.get_nowait()
        except queue.Empty:
            if not main_should_run:
                break
            time.sleep(1)
            continue

        client_id = job["client_id"]
        filename = job["name"]
        data = job["data"]
        scale = int(job["scale"])

        print("working", scale, client_id, filename)

        try:
        #if True:
            t1 = time.time()
            # Step 1: Load the image from bytes
            img = Image.open(BytesIO(data))
            img = img.convert("RGB")

            LQ_ips = img
            LQ_img, h0, w0 = PIL2Tensor(LQ_ips, scale, min_size=512)
            LQ_img = LQ_img.unsqueeze(0).to("cuda")[:, :3, :, :]

            captions = ['']

            model.to("cuda")
            # # step 3: Diffusion Process
            samples = model.batchify_sample(
                LQ_img,
                captions,
                num_steps=70,
                restoration_scale=-1,
                s_churn=5,
                s_noise=1.01,
                cfg_scale=4,
                control_scale=1,
                seed=random.randint(0,999999),
                num_samples=1,
                p_p='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.',
                n_p='painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth',
                color_fix_type='Wavelet',
                use_linear_CFG=True,
                use_linear_control_scale=False,
                cfg_scale_start=1,
                control_scale_start=0)

            model.to("cpu")
            torch.cuda.empty_cache()  # Frees unused memory
            torch.cuda.synchronize()  # Ensures all GPU operations finish
            # save
            upscaled = Tensor2PIL(samples[0], h0, w0)

            t2 = time.time()
            print("upscale with done in ", t2 - t1)

            if upscaled is not None:
                uploader_queue.put((client_id, filename, upscaled))
            else:
                # Handle error case
                error_data = {"client_id": client_id, "filename": filename}
                r = requests.post("/".join([server_url, "upscaling_delete_failed"]), data=error_data)
                print("Job failed, notified server:", r.status_code)

        except Exception as e:
        #if False:
            print("Job processing error:", e)
            error_data = {"client_id": client_id, "filename": filename}
            r = requests.post("/".join([server_url, "upscaling_delete_failed"]), data=error_data)
            print("Job failed, notified server:", r.status_code)

    uploader_should_run = False
    print("exiting main loop, waiting for uploader to join...")
    uploader_thread1.join()


if __name__ == "__main__":
    main()
