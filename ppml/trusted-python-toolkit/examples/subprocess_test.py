import subprocess

print("Running the pytorch task...")
subprocess.run(["python", "/ppml/examples/pytorch_test.py"], check=True)
print("Pytorch task completed.")

print("Running the numpy task...")
subprocess.run(["python", "/ppml/examples/numpy_test.py"], check=True)
print("Numpy task completed.")
