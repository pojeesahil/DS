import os

print("Soil Erosion Prediction System")
print("1) Train Model")
print("2) Predict Map")

choice = input("Enter choice (1/2): ")

if choice == "1":
    os.system("python train.py")
elif choice == "2":
    os.system("python predict.ipynb")
else:
    print("Invalid choice. Enter 1 or 2.")
