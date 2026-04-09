#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import os
import shutil
import sys
import warnings

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TOLERANCE = 0.5
RECURSIVE = True
# `None` birakirsan otomatik secilir.
CPUS = 4. #Kendi cihazina gore degistir
# Uzun kenari HD seviyesine indirir; buyuk fotograflarda ciddi hiz kazandirir.
MAX_DIMENSION = 1280
SHOW_LIVE_PROCESSING = True

WORKER_REFERENCE_ENCODINGS = None
WORKER_TOLERANCE = None
WORKER_MATCH_CONDITION = None

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


def import_face_recognition():
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError(
            "Bu script `face_recognition` ve `dlib` ister. "
            "Dogru conda env ile calistir: conda activate py310"
        ) from exc

    return face_recognition


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def ask_user_inputs():
    print("Secenekler:")
    print("1) Birden fazla yuzu ayni anda ara")
    print("2) Tek bir yuz ara")
    
    secim = input("Lutfen seciminizi yapin (1/2) [Varsayilan: 2]: ").strip()
    if secim not in ["1"]:
        secim = "2"
        
    reference_images = []
    match_condition = "biri"
    
    if secim == "1":
        print("\nBirden fazla referans yuzu nasil vermek istersiniz?")
        print("- Bir klasor yolu verebilirsiniz (Klasordeki tum fotograflar referans sayilir, ornek: grup_photos)")
        print("- Veya gorsel yollarini virgulle ayirarak yazabilirsiniz (ornek: yuz1.jpg, yuz2.png)")
        girdi = input("Klasor yolu VEYA virgulle ayrilmis yollar [Varsayilan: grup_photos]: ").strip()
        
        if not girdi:
            girdi = "grup_photos"
            
        girdi = girdi.strip('\'" ')

        if os.path.isdir(girdi):
            print(f"'{girdi}' klasorundeki gorseller taranarak referans olarak eklenecek.")
            for file_name in os.listdir(girdi):
                file_path = os.path.join(girdi, file_name)
                if os.path.isfile(file_path) and is_image_file(file_path):
                    reference_images.append(file_path)
            if not reference_images:
                print(f"Uyari: '{girdi}' klasorunde gorsel bulunamadi!")
        else:
            paths = [p.strip().strip('\'" ') for p in girdi.split(",")]
            for p in paths:
                if p:
                    reference_images.append(p)

        if len(reference_images) > 1:
            while True:
                condition = input("Hedef fotograflarda bu kisilerden EN AZ BIRI olmasi yeterli mi, yoksa HEPSI birden mi bulunmali? (Biri / Hepsi) [Varsayilan: Biri]: ").strip().lower()
                if not condition:
                    condition = "biri"
                if condition in ["biri", "hepsi"]:
                    match_condition = condition
                    break
                print("Lutfen 'biri' veya 'hepsi' olarak cevaplayiniz.")
    else:
        girdi = input("Referans fotografin yolunu giriniz [Varsayilan: reference.jpeg]: ").strip()
        if not girdi:
            girdi = "reference.jpeg"
        girdi = girdi.strip('\'" ')
        reference_images.append(girdi)

    source_folder = input("Taranacak klasorun (hedef dosyalar) yolunu giriniz: ").strip().strip('\'" ')
    output_folder = input("Cikis klasorunun yolunu giriniz: ").strip().strip('\'" ')

    return (
        reference_images,
        match_condition,
        source_folder,
        output_folder,
        TOLERANCE,
        RECURSIVE,
        CPUS,
        MAX_DIMENSION,
        SHOW_LIVE_PROCESSING,
    )


def load_image_for_face_recognition(image_path, max_dimension):
    image = Image.open(image_path)
    image = image.convert("RGB")

    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    return np.array(image)


def load_reference_encoding(reference_image, max_dimension):
    face_recognition = import_face_recognition()
    image = load_image_for_face_recognition(reference_image, max_dimension)
    locations = face_recognition.face_locations(image, model="hog")

    if not locations:
        raise ValueError(f"Fotografta yuz bulunamadi.")

    if len(locations) > 1:
        print(f"Uyari: Referans olarak verilen '{reference_image}' icinde birden fazla yuz bulundu. Ilk yuz kullanilacak.")

    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=[locations[0]],
    )
    return encodings[0]


def iter_image_files(source_folder, recursive):
    if recursive:
        for root, _, files in os.walk(source_folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if is_image_file(file_path):
                    yield file_path
    else:
        for file_name in os.listdir(source_folder):
            file_path = os.path.join(source_folder, file_name)
            if os.path.isfile(file_path) and is_image_file(file_path):
                yield file_path


def copy_match(source_path, source_folder, output_folder):
    relative_path = os.path.relpath(source_path, source_folder)
    destination_path = os.path.join(output_folder, relative_path)

    if os.path.exists(destination_path):
        return destination_path, True

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path, False


def resolve_worker_count(requested_cpus):
    available_cpus = os.cpu_count() or 1
    if requested_cpus is None:
        return max(1, min(available_cpus - 1, 8))

    return max(1, min(int(requested_cpus), available_cpus))


def init_worker(reference_encodings, tolerance, match_condition):
    global WORKER_REFERENCE_ENCODINGS
    global WORKER_TOLERANCE
    global WORKER_MATCH_CONDITION
    WORKER_REFERENCE_ENCODINGS = reference_encodings
    WORKER_TOLERANCE = tolerance
    WORKER_MATCH_CONDITION = match_condition


def scan_single_image(task):
    image_index, total_images, image_path, show_live_processing = task
    try:
        if show_live_processing:
            print(f"[{image_index}/{total_images}] Isleniyor: {image_path}", flush=True)

        face_recognition = import_face_recognition()
        image = load_image_for_face_recognition(image_path, MAX_DIMENSION)
        locations = face_recognition.face_locations(image, model="hog")

        if not locations:
            return image_path, False, None

        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=locations,
        )

        if WORKER_MATCH_CONDITION == "biri":
            # Herhangi bir referans yüzü bulunursa eslestir
            for encoding in encodings:
                for ref_encoding in WORKER_REFERENCE_ENCODINGS:
                    is_match = face_recognition.compare_faces(
                        [ref_encoding],
                        encoding,
                        tolerance=WORKER_TOLERANCE,
                    )[0]
                    if is_match:
                        return image_path, True, None
            return image_path, False, None

        else:
            # "hepsi" kosulu: tum referans yuzlerinin fotografta olmasi lazim
            matched_refs = set()
            for encoding in encodings:
                for ref_i, ref_encoding in enumerate(WORKER_REFERENCE_ENCODINGS):
                    if ref_i in matched_refs:
                        continue
                    is_match = face_recognition.compare_faces(
                        [ref_encoding],
                        encoding,
                        tolerance=WORKER_TOLERANCE,
                    )[0]
                    if is_match:
                        matched_refs.add(ref_i)
                        break
            
            # Bulunan farkli referans yuzu sayisi, toplam aradigimiz referans sayisina esitse onay ver
            if len(matched_refs) == len(WORKER_REFERENCE_ENCODINGS):
                return image_path, True, None
            
            return image_path, False, None

    except Exception as exc:
        return image_path, False, str(exc)


def process_images_parallel(image_paths, reference_encodings, tolerance, match_condition, worker_count):
    start_method = "spawn"
    if "forkserver" in multiprocessing.get_all_start_methods():
        start_method = "forkserver"

    context = multiprocessing.get_context(start_method)
    chunksize = max(1, len(image_paths) // (worker_count * 4)) if image_paths else 1

    pool = context.Pool(
        processes=worker_count,
        initializer=init_worker,
        initargs=(reference_encodings, tolerance, match_condition),
    )
    try:
        for result in pool.imap_unordered(scan_single_image, image_paths, chunksize=chunksize):
            yield result
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise


def main():
    (
        reference_images,
        match_condition,
        source_folder,
        output_folder,
        tolerance,
        recursive,
        requested_cpus,
        max_dimension,
        show_live_processing,
    ) = ask_user_inputs()

    if not reference_images:
        print("Hata: Gecerli bir referans fotograf bulunamadi.")
        return 1

    for ref_img in reference_images:
        if not os.path.isfile(ref_img):
            print(f"Hata: referans fotograf bulunamadi: {ref_img}")
            return 1

    if not source_folder:
        print("Hata: Lutfen taranacak klasor yolunu giriniz.")
        return 1
        
    if not os.path.isdir(source_folder):
        print("Hata: taranacak klasor bulunamadi.")
        return 1

    if not output_folder:
        print("Hata: Lutfen cikis klasoru yolunu giriniz.")
        return 1
        
    os.makedirs(output_folder, exist_ok=True)

    reference_encodings = []
    print("\nReferans yuzler isleniyor...")
    for ref_img in reference_images:
        try:
            encoding = load_reference_encoding(ref_img, max_dimension)
            reference_encodings.append(encoding)
            print(f" -> Basarili: {os.path.basename(ref_img)}")
        except Exception as exc:
            print(f"Hata: referans fotograf islenemedi ({ref_img}): {exc}")
            return 1

    image_paths = list(iter_image_files(source_folder, recursive))
    if not image_paths:
        print("Hata: taranacak hedef fotograf bulunamadi.")
        return 1

    worker_count = resolve_worker_count(requested_cpus)
    tasks = [
        (index, len(image_paths), image_path, show_live_processing)
        for index, image_path in enumerate(image_paths, start=1)
    ]
    matched_count = 0
    copied_count = 0
    skipped_existing_count = 0
    error_count = 0

    print("\n--------------------------")
    print(f"Toplam hedef fotograf: {len(image_paths)}")
    print(f"Aranan referans yuz sayisi: {len(reference_encodings)}")
    if len(reference_encodings) > 1:
        match_text = "EN AZ BIRI" if match_condition == "biri" else "HEPSI"
        print(f"Eslesme kurali: Fotografta referans yuzlerden {match_text} bulunmali.")
    print(f"Kullanilan islemci: {worker_count}")
    print(f"Maksimum uzun kenar: {max_dimension}px")
    print("--------------------------\n")

    try:
        for scanned_count, (image_path, is_match, error_message) in enumerate(
            process_images_parallel(
                tasks,
                reference_encodings,
                tolerance,
                match_condition,
                worker_count,
            ),
            start=1,
        ):
            if error_message:
                error_count += 1
                print(f"Atlandi: {image_path} ({error_message})", file=sys.stderr)
                continue

            if is_match:
                matched_count += 1
                destination_path, skipped_existing = copy_match(
                    image_path,
                    source_folder,
                    output_folder,
                )

                if skipped_existing:
                    skipped_existing_count += 1
                    print(f"[{scanned_count}/{len(image_paths)}] Zaten var: {destination_path}")
                else:
                    copied_count += 1
                    print(f"[{scanned_count}/{len(image_paths)}] Eslesme bulundu: {destination_path}")
            elif scanned_count % 10 == 0 or scanned_count == len(image_paths):
                print(f"[{scanned_count}/{len(image_paths)}] Tarama devam ediyor...")
    except KeyboardInterrupt:
        print("\nTarama kullanici tarafindan durduruldu.")
        return 130

    print("")
    print(f"Taranan fotograf sayisi: {len(image_paths)}")
    print(f"Eslesen fotograf sayisi: {matched_count}")
    print(f"Yeni kopyalanan fotograf sayisi: {copied_count}")
    print(f"Mevcut oldugu icin atlanan fotograf sayisi: {skipped_existing_count}")
    print(f"Hata nedeniyle atlanan fotograf sayisi: {error_count}")
    print(f"Cikti klasoru: {os.path.abspath(output_folder)}")

    if matched_count == 0:
        print("Referans yuzlerle eslesen hedef fotograf bulunamadi.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
