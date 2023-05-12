
<p align="center">
	<img height="500px" src="illustration/rigaa_robot.png"/>
</p>
<h1 align="center">
	RL based maze generation
</h1>

## Usage
1. Intall all the requirements:
```
pip install requirements.txt
```
2. Run the training:
```
python train.py
```
Training can take a long time.
Add this line to the step() function of the environemt if you want to see intermediate steps.
```
self.render()
```
3. Evaluate the produced model:
```
python evaluate.py
```