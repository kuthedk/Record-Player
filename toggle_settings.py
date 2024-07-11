import os


def toggle_processing_settings(config):
    print("\nToggle Processing Settings:")
    print("1: Toggle De-essing")
    print("2: Toggle Stereo Enhancement")
    print("3: Toggle Soft Clipping")
    print("4: Toggle Click and Pop Removal")
    print("5: Toggle Noise Reduction")
    print("6: Toggle Equalization")
    print("7: Toggle Phase Correction")
    print("8: Toggle Wow and Flutter Correction")
    print("9: Toggle Harmonic Excitation")
    print("0: Toggle Dynamic Range Expansion")
    print("Enter your choice: ", end="")
    choice = input().strip()

    if choice == "1":
        config.ENABLE_DE_ESSING = not config.ENABLE_DE_ESSING
        print(f"De-essing {'enabled' if config.ENABLE_DE_ESSING else 'disabled'}")
    elif choice == "2":
        config.ENABLE_STEREO_ENHANCEMENT = not config.ENABLE_STEREO_ENHANCEMENT
        print(
            f"Stereo enhancement {'enabled' if config.ENABLE_STEREO_ENHANCEMENT else 'disabled'}"
        )
    elif choice == "3":
        config.ENABLE_SOFT_CLIPPING = not config.ENABLE_SOFT_CLIPPING
        print(
            f"Soft clipping {'enabled' if config.ENABLE_SOFT_CLIPPING else 'disabled'}"
        )
    elif choice == "4":
        config.ENABLE_CLICK_POP_REMOVAL = not config.ENABLE_CLICK_POP_REMOVAL
        print(
            f"Click and pop removal {'enabled' if config.ENABLE_CLICK_POP_REMOVAL else 'disabled'}"
        )
    elif choice == "5":
        config.ENABLE_NOISE_REDUCTION = not config.ENABLE_NOISE_REDUCTION
        print(
            f"Noise reduction {'enabled' if config.ENABLE_NOISE_REDUCTION else 'disabled'}"
        )
    elif choice == "6":
        config.ENABLE_EQUALIZATION = not config.ENABLE_EQUALIZATION
        print(f"Equalization {'enabled' if config.ENABLE_EQUALIZATION else 'disabled'}")
    elif choice == "7":
        config.ENABLE_PHASE_CORRECTION = not config.ENABLE_PHASE_CORRECTION
        print(
            f"Phase correction {'enabled' if config.ENABLE_PHASE_CORRECTION else 'disabled'}"
        )
    elif choice == "8":
        config.ENABLE_WOW_FLUTTER_CORRECTION = not config.ENABLE_WOW_FLUTTER_CORRECTION
        print(
            f"Wow and flutter correction {'enabled' if config.ENABLE_WOW_FLUTTER_CORRECTION else 'disabled'}"
        )
    elif choice == "9":
        config.ENABLE_HARMONIC_EXCITATION = not config.ENABLE_HARMONIC_EXCITATION
        print(
            f"Harmonic excitation {'enabled' if config.ENABLE_HARMONIC_EXCITATION else 'disabled'}"
        )
    elif choice == "0":
        config.ENABLE_DYNAMIC_RANGE_EXPANSION = (
            not config.ENABLE_DYNAMIC_RANGE_EXPANSION
        )
        print(
            f"Dynamic range expansion {'enabled' if config.ENABLE_DYNAMIC_RANGE_EXPANSION else 'disabled'}"
        )
    else:
        print("Invalid choice")


def print_processing_settings(config):
    os.system("clear")
    print("Current processing settings:")
    print(f"  De-essing: {'Enabled' if config.ENABLE_DE_ESSING else 'Disabled'}")
    print(
        f"  Stereo Enhancement: {'Enabled' if config.ENABLE_STEREO_ENHANCEMENT else 'Disabled'}"
    )
    print(
        f"  Soft Clipping: {'Enabled' if config.ENABLE_SOFT_CLIPPING else 'Disabled'}"
    )
    print(
        f"  Click and Pop Removal: {'Enabled' if config.ENABLE_CLICK_POP_REMOVAL else 'Disabled'}"
    )
    print(
        f"  Noise Reduction: {'Enabled' if config.ENABLE_NOISE_REDUCTION else 'Disabled'}"
    )
    print(f"  Equalization: {'Enabled' if config.ENABLE_EQUALIZATION else 'Disabled'}")
    print(
        f"  Phase Correction: {'Enabled' if config.ENABLE_PHASE_CORRECTION else 'Disabled'}"
    )
    print(
        f"  Wow and Flutter Correction: {'Enabled' if config.ENABLE_WOW_FLUTTER_CORRECTION else 'Disabled'}"
    )
    print(
        f"  Harmonic Excitation: {'Enabled' if config.ENABLE_HARMONIC_EXCITATION else 'Disabled'}"
    )
    print(
        f"  Dynamic Range Expansion: {'Enabled' if config.ENABLE_DYNAMIC_RANGE_EXPANSION else 'Disabled'}"
    )
    print("\nPress 't' to toggle processing settings.")
