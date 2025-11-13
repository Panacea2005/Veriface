"""
Test script to verify preprocessing matches training notebook exactly.
Run: python test_preprocessing.py
"""
import numpy as np

def notebook_preprocessing(pixel_value):
    """
    Notebook training preprocessing:
    transforms.Lambda(lambda x:(x*255-127.5)/128.0)
    where x is in [0, 1] from ToTensor()
    """
    # ToTensor converts [0, 255] -> [0, 1]
    normalized = pixel_value / 255.0
    # Apply lambda
    result = (normalized * 255.0 - 127.5) / 128.0
    return result

def backend_old_preprocessing(pixel_value):
    """OLD backend preprocessing (WRONG)"""
    normalized = pixel_value / 255.0
    result = (normalized - 0.5) / 0.5
    return result

def backend_new_preprocessing(pixel_value):
    """NEW backend preprocessing (CORRECT)"""
    result = (pixel_value - 127.5) / 128.0
    return result

# Test with key pixel values
test_pixels = [0, 64, 127, 128, 192, 255]

print("=" * 80)
print("PREPROCESSING VERIFICATION TEST")
print("=" * 80)
print(f"\n{'Pixel':<10} {'Notebook':<15} {'Backend OLD':<15} {'Backend NEW':<15} {'Match?':<10}")
print("-" * 80)

all_match = True
for pixel in test_pixels:
    nb_result = notebook_preprocessing(pixel)
    be_old_result = backend_old_preprocessing(pixel)
    be_new_result = backend_new_preprocessing(pixel)
    
    # Check if new backend matches notebook (within floating point tolerance)
    matches = abs(nb_result - be_new_result) < 1e-6
    all_match = all_match and matches
    
    match_symbol = "✓" if matches else "✗"
    print(f"{pixel:<10} {nb_result:<15.6f} {be_old_result:<15.6f} {be_new_result:<15.6f} {match_symbol:<10}")

print("-" * 80)

# Show the difference between old and new
print("\nDifference between OLD and NEW backend:")
for pixel in test_pixels:
    old = backend_old_preprocessing(pixel)
    new = backend_new_preprocessing(pixel)
    diff = abs(old - new)
    print(f"  Pixel {pixel:3d}: diff = {diff:.6f} ({diff/new*100:.2f}% relative error)")

print("\n" + "=" * 80)
if all_match:
    print("✓ SUCCESS: Backend NEW preprocessing matches notebook EXACTLY!")
    print("  Model will now have consistent preprocessing between training and inference.")
else:
    print("✗ FAILURE: Preprocessing mismatch detected!")
    print("  This will cause significant accuracy degradation.")
print("=" * 80)

# Mathematical proof
print("\nMATHEMATICAL PROOF:")
print("-" * 80)
print("Notebook: (x/255 * 255 - 127.5) / 128.0 = (x - 127.5) / 128.0")
print("Backend NEW: (x - 127.5) / 128.0")
print("✓ These are IDENTICAL")
print()
print("Backend OLD: ((x/255) - 0.5) / 0.5 = (x/255 - 0.5) / 0.5")
print("           = x/(255*0.5) - 0.5/0.5 = x/127.5 - 1")
print("           = (x - 127.5) / 127.5")
print("✗ This uses 127.5 as divisor instead of 128.0 -> WRONG!")
print("-" * 80)

# Range verification
print("\nOUTPUT RANGE VERIFICATION:")
print("-" * 80)
print(f"Notebook [0, 255] -> [{notebook_preprocessing(0):.6f}, {notebook_preprocessing(255):.6f}]")
print(f"Backend NEW [0, 255] -> [{backend_new_preprocessing(0):.6f}, {backend_new_preprocessing(255):.6f}]")
print(f"Backend OLD [0, 255] -> [{backend_old_preprocessing(0):.6f}, {backend_old_preprocessing(255):.6f}]")
print()
print("Note: Notebook uses /128.0 which gives range ≈[-0.996, 0.996]")
print("      Standard normalization uses /127.5 which gives range [-1.0, 1.0]")
print("      This small difference (0.4%) can impact model accuracy!")
print("=" * 80)
