# Quick test script
import zipfile
with zipfile.ZipFile("C:/Users/athar/Downloads/eyedentifySR.zip", 'r') as z:
    # Test if the corrected path exists
    test_path = "eyedentify++/RealEsr_x2/left_eyes/24/24/frame_01.png"
    try:
        with z.open(test_path):
            print(f"SUCCESS: Found {test_path}")
    except KeyError:
        print(f"FAILED: Still can't find {test_path}")
        # List what's actually there
        files = [f for f in z.namelist() if 'left_eyes/24/24/' in f and 'frame' in f]
        print("Available files:", files[:5])
