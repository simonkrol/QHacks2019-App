# UrbanSound - QHacks2019 Project - [Github](https://github.com/simonkrol/QHacks2019-app)

### Team Members:
- [Simon Krol](https://github.com/simonkrol)

## Setup
Create a virtual environment to house the project (optional):
```
virtualenv -p python3 UrbanSound
cd UrbanSound
```
Clone the repository:
```
git clone https://github.com/simonkrol/QHacks2019-app.git
```
Install dependencies:
```
pip install -r requirements.txt
```
Since the model is too large to host on GitHub, it is not included in the repository. If you'd like, you can download the model I trained, from my [Google Drive](https://drive.google.com/open?id=1z7WXMsFGwOoIOmkicSlqo5g5qnyq06ft) (I've also included a few test files)

## Usage
Start the server:
```
python app.py
```
Either send a request through the requests library, postman or access the webpage endpoint at `http://localhost:5000`

The result values are expressed as a list of Strings containing the events occuring throughout each part of the sound file.
```
Sirens from 0 until 3 seconds
Dog Barking from 4 until 9 seconds
No Identifiable Sound from 9 until 12 seconds
Gun Shot at 13 seconds
```
