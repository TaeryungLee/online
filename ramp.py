def ramp(t, w, W: int, k: int):
    """
    Ramp: r = 1 - ((w - tk - k) / (W - 2k - 1))
    - t: [B,1] in [0,1] (continuous) or [B] (자동 확장)
    - w: [1,W] (0..W-1)
    반환: [B,W] in [0,1]
    """
    return 1 - ((w - t*(k + 1) - k) / (W - 2*k - 1))

print(ramp(0, 0, 64, 7))
print(ramp(0, 7, 64, 7))
print(ramp(0, 8, 64, 7))
print(ramp(0, 55, 64, 7))
print(ramp(0, 56, 64, 7))
print(ramp(0, 63, 64, 7))

print()
print(ramp(1, 0, 64, 7))
print(ramp(1, 15, 64, 7))
print(ramp(1, 16, 64, 7))
print(ramp(1, 63, 64, 7))

