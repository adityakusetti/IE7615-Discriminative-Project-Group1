# IE7615 Discriminative Project – Group 1

Aditi Kulkarni, Sakshi Pawar, Aditya Sai Kumar Kusetti, Hemanth Mareedu

Object detection with YOLOv8 on the class dataset (73 objects, OBJ001 to OBJ073). Given an image, the model finds every object in it and tells us its Object ID and where it is.

## How it works

**1. The problem with the dataset**
The shared dataset only has photos of one object each (about 95 per object, all 224x224). A detector needs images with several objects and a box around each one, so we had to make those ourselves.

**2. Making multi-object images**
We take random photos from different objects and paste them into a grid: 2x2, 3x3, and a few 2x3 and 3x2. Since we decide which cell each photo goes in, we already know its box, so the labels are created automatically and nothing is drawn by hand. Some grids leave a cell or two empty so the number of objects changes. For Milestone 3 we also added single photos (1x1) so the model can handle one object on its own.

We removed the background-only photos from Milestone 1 because they aren't needed for detection.

**3. Splitting the data**
Before building any grids we split the single photos into train (70%), validation (15%) and test (15%). This way a photo used in a training grid never shows up in a test grid, so the test score is fair.

**4. Training**
We fine-tuned YOLOv8s (pretrained on COCO) for up to 60 epochs with early stopping, image size 672 (the size of a full 3x3 grid), on a Colab T4 GPU. Training took about 20–25 minutes.

**5. Single objects**
Passing a single photo straight in only got 82.7% right, because the model mostly learned objects at the size of a grid tile. Placing the photo in the corner of a 2x2-sized gray image, like one tile of a grid, raised it to 96.7%. The demo does this automatically.

## Results

| | Milestone 2 | Milestone 3 (final) |
|---|---|---|
| Training images | 1,600 | 2,000 |
| Test mAP50 | 0.982 | 0.972 |
| Test precision / recall | 0.974 / 0.960 | 0.981 / 0.942 |
| Single-object accuracy | – | 96.7% |

## Files

- `IE7615_DiscriminativeProject_Milestone2_Group1.ipynb` – Milestone 2: building the grids, training and testing
- `IE7615_DiscriminativeProject_Milestone3_Group1.ipynb` – Milestone 3: final model with single photos added
- `IE7615_Milestone3_Demo.ipynb` – demo notebook
- `best.pt` – trained model (Milestone 3)

## Running the demo

1. Open `IE7615_Milestone3_Demo.ipynb` in Google Colab.
2. Run the cells. The model is downloaded from this repo.
3. Upload an image (folder icon on the left) and run:

```python
detect("/content/image.jpg", single=True)   # one object -> prints its Object ID
detect("/content/image.jpg")                # grid -> prints every Object ID and its box
```
