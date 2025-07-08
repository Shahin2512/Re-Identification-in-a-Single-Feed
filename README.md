# ⚽ Player Re-Identification in a Single Feed

This project uses a fine-tuned YOLOv11 model and Deep SORT tracker to detect and re-identify players in football match footage. Even if a player leaves the frame and returns, the system maintains a consistent ID for them across the entire video.

**Goal:**  
- Detect and track only `players` across the video.
- Assign unique IDs to each player.
- Maintain consistent IDs even if a player leaves and re-enters the frame.

---

## 🚀 Tech Stack

| Component       | Technology                  |
|----------------|-----------------------------|
| Object Detection | [Ultralytics YOLOv11 (PyTorch)] |
| Tracking         | [Deep SORT] |
| Video I/O        | OpenCV                      |
| Language         | Python 3.10                 |

## 📁 Project Structure

Re-Identification in a Single Feed/
├── best.pt # YOLOv11 model (not committed in Git)
├── inputs/ # Folder for raw input videos (.mp4)
│ ├── input1.mp4
│ ├── input2.mp4
│ └── input3.mp4
├── outputs/ # Folder for annotated/tracked videos
├── track_players.py # Main script
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This file

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repo

```bash
git clone <your-repo-url>
cd Re-Identification in a Single Feed

 ## How to Run the Code
python track_players.py

Check the outputs/ folder for results
