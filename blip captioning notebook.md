# BLIP ile Görselden Metin Üretme (Captioning) Pipeline

## ✅ 1. Gerekli Kütüphanelerin Kurulumu ve Yüklenmesi
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import torch
import os
```

## ✅ 2. Model ve Processor’ü Yükleme
```python
model_path = "/kaggle/input/blip-model-local"

processor = BlipProcessor.from_pretrained(f"{model_path}/blip-model")
model = BlipForConditionalGeneration.from_pretrained(f"{model_path}/blip-model")

# Cihaz ataması
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## ✅ 3. Verilerin Yüklenmesi ve Dönüştürülmesi
```python
from PIL import Image
import os

def transform(example):
    image_path = os.path.join("/kaggle/input/train-dataset/train", str(example["image_id"]) + ".jpg")
    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        images=image,
        text=example["caption"],
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    return {
        "pixel_values": encoding["pixel_values"][0],
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0]
    }
```

## ✅ 4. Dataset Üzerinde Dönüşüm
```python
dataset = load_dataset("csv", data_files="/kaggle/input/train-metadata/train.csv")
processed_data = dataset["train"].map(transform)
```

## ✅ 5. Dataloader ve Optimizer Tanımı
```python
from torch.utils.data import DataLoader
from torch.optim import AdamW

train_loader = DataLoader(processed_data, batch_size=8, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-5)
```

## ✅ 6. Model Eğitimi
```python
from tqdm import tqdm
import gc

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in tqdm(train_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        
        del pixel_values, input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")
```

## ✅ 7. Modeli Kaydetme
```python
model.save_pretrained("/kaggle/working/blip-custom-model")
processor.save_pretrained("/kaggle/working/blip-custom-processor")
```

## ✅ 8. Test Seti ile Tahmin Üretme
```python
import pandas as pd
from tqdm import tqdm

# Test verisi
test_df = pd.read_csv("/kaggle/input/test-data/test.csv")

results = []
for image_id in tqdm(test_df["image_id"]):
    image_path = f"/kaggle/input/test-data/test/test/{image_id}.jpg"
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    results.append({"image_id": image_id, "caption": caption})
```

## ✅ 9. Submission Dosyası Kaydetme
```python
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)
