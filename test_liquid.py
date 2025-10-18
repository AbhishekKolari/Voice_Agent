# Test script - save as test_liquid.py
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    print("✓ Liquid audio imported successfully")
    
    # Try loading with CPU
    processor = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2-Audio-1.5B").eval()
    print("✓ Processor loaded")
    
    print("Loading model...")
    model = LFM2AudioModel.from_pretrained("LiquidAI/LFM2-Audio-1.5B").eval()
    # torch.cuda.empty_cache()
    print("✓ Model loaded")
    # model = LFM2AudioModel.from_pretrained(
    #     "LiquidAI/LFM2-Audio-1.5B", 
    #     device="cpu"  # Use the supported 'device' argument
    # )
    # print("✓ Model loaded onto CPU.")
    
    # # 2. Manually convert to FP16 and move to GPU
    # # This halves the memory footprint for the VRAM transfer.
    # model = model.half().to('cuda') 
    # print("✓ Model successfully moved to GPU in FP16.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()



# # test_cuda.py
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# # Test CUDA works
# if torch.cuda.is_available():
#     x = torch.zeros(1).cuda()
#     print("✓ CUDA test successful!")
# else:
#     print("✗ CUDA not available")