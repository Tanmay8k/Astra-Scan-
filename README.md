
# 🌕 ASTRA-SCAN
### AI-Powered Lunar Rover Navigation & Mission Control System

*Making Lunar Exploration Smarter, Safer, and More Autonomous.*

---

## 🚀 About the Project

ASTRA-SCAN is an AI-powered, browser-based rover operating and ground-control platform designed for exploring the challenging regions of the Lunar South Pole.

The project focuses on helping a lunar rover understand its surroundings, identify dangerous terrain, choose safer routes, manage its energy, and handle communication failures.

Using real ISRO Chandrayaan-3 mission data, machine learning, 3D terrain visualization, and autonomous navigation techniques, ASTRA-SCAN brings important rover operations together in one interactive dashboard.

Our goal is simple: *to make lunar exploration safer, smarter, and more reliable, even in challenging environments.*

## 🌑 Problem Statement

Exploring the Lunar South Pole comes with several challenges:

- *Difficult Terrain:* Steep crater walls and uneven surfaces can make rover movement dangerous.
- *Extreme Temperatures:* Permanently Shadowed Regions (PSRs) can experience temperatures below 105 Kelvin, creating harsh conditions and possible water-ice regions.
- *Limited Sunlight:* Low-angle sunlight creates long shadows and makes solar energy management difficult.
- *Communication Blackouts:* Craters and terrain obstacles can block direct communication between the rover and Earth.
- *Navigation Risks:* A rover needs to identify safe paths without depending entirely on continuous instructions from Earth.

Traditional remote-controlled approaches may not be sufficient in such conditions. A smarter system is needed that can analyse the terrain and make safer decisions independently.

## 💡 Our Solution

ASTRA-SCAN combines AI-based terrain analysis, autonomous pathfinding, 3D lunar mapping, energy monitoring, and communication recovery.

It uses available lunar datasets to understand the environment and provides an interactive mission-control dashboard where users can monitor rover activities, explore terrain, test different conditions, and manage emergency situations.

## ✨ Key Features

### 1. Automated ISRO Dataset Integration
- Automatically extracts uploaded ISRO dataset archives.
- Reads PDS4 metadata.
- Processes and organises calibrated lunar data.
- Stores dataset information in an SQLite database.
- Provides a dedicated Chandrayaan Datasets section.

### 2. AI-Based Terrain Classification
Our machine learning model analyses:
- Surface elevation
- Terrain slope
- Surface roughness
- ChaSTE temperature readings

It identifies four important terrain categories:

- *Safe Regolith:* Areas suitable for rover movement.
- *Crater Wall Hazards:* Steep regions that may exceed safe rover tilt limits.
- *Potential Cold Traps:* Regions below 105 Kelvin where water ice may exist.
- *Sunlit High-Elevation Peaks:* Areas useful for solar charging and communication.

### 3. Interactive 3D Lunar Visualization

ASTRA-SCAN provides a 3D view of the lunar surface using a 96 × 96 terrain grid, with each cell representing 10 metres.

Important lunar locations include:
- Shiv Shakti Point
- Shackleton Crater
- Cabeus Cold Trap
- Malapert Massif

*Three visualization modes:*
- Model 1 – Full spherical Moon view.
- Model 2 – Local 440 × 440 metre terrain view.
- Model 3 – Morph mode connecting spherical and local terrain views.

Users can also switch between elevation maps, ChaSTE Cryogenic Thermal Heatmaps, and AI South Pole Geology layers.

### 4. What-If Mission Simulator

The What-If feature allows users to experiment with different mission conditions by changing:
- Sun elevation angle.
- Maximum safe rover tilt limit.

This helps demonstrate how environmental changes can affect rover movement and safety.

### 5. Dual Camera and Rover Vision System

The dashboard includes two camera panels:

- *CAM 1:* Displays Pragyan NavCam imagery or a connected live webcam, with object detection, targeting HUD, and estimated object dimensions.
- *CAM 2:* Displays a synchronized ChaSTE-based thermal horizon view.

These features help users understand the rover's surroundings.

### 6. Autonomous Navigation and A* Pathfinding

The 60-metre lookahead system checks three possible movement directions:
- Left
- Centre
- Right

It recommends whether the rover should continue, change direction, or take another route.

The AI 60m Scan Step uses A* pathfinding to guide the rover towards safe and unexplored terrain.

### 7. Smart Energy Management

ASTRA-SCAN includes:
- Real-time power telemetry in watts.
- Solar Surfing for energy-aware movement and charging.
- Eco Power Mode to reduce energy consumption.
- Thruster Jump for controlled movement across difficult terrain.

### 8. Communication Blackout and SAFETY BACKUP

Communication loss is one of the major challenges of lunar exploration.

ASTRA-SCAN addresses this through:

- *Movement Tracking:* Records rover movements in a timestamped SQLite breadcrumb trail.
- *SAFETY BACKUP:* Activates backup readiness when communication is lost.
- *South Pole Vault:* Saves captured photos locally during a blackout.
- *Unsynced Data Storage:* Marks photos that are waiting to be synchronized.
- *Retrace Backup:* Allows the rover to follow its last 10-minute movement history backwards.
- *Automatic Synchronization:* Transfers queued photos and data once communication is restored.

### 9. Mission Safety and Logs

The Command Deck provides three important controls:

- *Mission Abort:* Immediately locks rover propulsion during an emergency.
- *Clear Abort & Resume:* Allows operations to resume after an abort condition is cleared.
- *Full Mission Reset:* Resets the mission system.

The Mission Logs section maintains a record of important system events, movements, and safety actions.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| React.js | Frontend and dashboard interface |
| Vite | Frontend development and build tool |
| Tailwind CSS | Dashboard styling |
| Python | Backend logic and data processing |
| Flask | Backend API |
| NumPy | Numerical data processing |
| Rasterio | Geospatial and terrain data handling |
| Scikit-learn | Machine learning and terrain classification |
| Plotly / Three.js | Data visualization and 3D terrain |
| SQLite | Mission logs, datasets and local storage |
| OpenCV / Webcam API | Camera and image processing |

## ⚙️ How ASTRA-SCAN Works

The overall workflow can be understood in six steps:

1. *Data Collection:* ISRO Chandrayaan-3 datasets are uploaded and extracted.
2. *Data Processing:* PDS4 metadata and calibrated sensor information are processed and stored.
3. *Terrain Analysis:* The machine learning model identifies safe and hazardous lunar regions.
4. *Navigation:* A* pathfinding and lookahead analysis help select safer movement paths.
5. *Mission Monitoring:* The dashboard displays 3D terrain, camera views, telemetry, and rover activities.
6. *Recovery:* In case of communication loss, local storage and breadcrumb-based retracing help protect mission data and support recovery.

## 💻 Installation and Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or above
- Node.js and npm
- Visual Studio Code
- Git

### 1. Clone the Repository

bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ASTRA-SCAN


### 2. Set Up the Frontend

bash
npm install
npm run dev


Open the local URL displayed in your terminal to access the dashboard.

### 3. Set Up the Python Backend

Create and activate a virtual environment:

bash
python -m venv venv


On Windows:

bash
venv\Scripts\activate


Install the required Python packages:

bash
pip install -r requirements.txt


Start the Flask backend using your project's configured entry-point file.

Note: The exact backend start command and environment configuration should match the final repository structure.

## 🔮 Future Improvements

We aim to improve ASTRA-SCAN further through:

- More advanced AI-based hazard prediction.
- Improved real-time lunar terrain analysis.
- Better energy-efficient autonomous navigation.
- More accurate communication blackout prediction.
- Integration with additional ISRO and international lunar datasets.
- Further development towards real rover hardware integration.

## 👥 Our Team

*ASTRA-SCAN Development Team*

- Jharna Gupta
- Tanmay
- Arkabrata
- Yashneil

## 🌍 Our Vision

We believe that future space exploration will depend on intelligent systems that can understand their surroundings and make decisions independently.

ASTRA-SCAN is our step towards that vision, combining artificial intelligence, lunar science, and autonomous rover operations into one unified platform.

*Explore Smarter. Navigate Safer. Discover Further.*

---

## 🙌 Acknowledgements

We acknowledge ISRO's Chandrayaan-3 mission and its scientific datasets, which provide the foundation for our lunar exploration project.

We also acknowledge the open-source technologies and tools that support the development of ASTRA-SCAN.

---

Built with curiosity, teamwork, and a vision for the future of space exploration. 🚀
