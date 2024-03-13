# Install GPU Driver on Linux

This guide demonstrates how to install Intel GPU driver on linux with kernel version 5.19/

It applies to Intel Arc Series GPU.

* install arc driver
    ```bash
    sudo apt-get install -y gpg-agent wget
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
    ```

* downgrade kernel
    ```bash
    sudo apt-get update && sudo apt-get install  -y --install-suggests  linux-image-5.19.0-41-generic

    sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1> $(echo $(($(awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg \
    | grep -no '5.19.0-41' | sed 's/:/\n/g' | head -n 1)-2)))\"/" /etc/default/grub

    sudo  update-grub

    sudo reboot
    # As 5.19's kernel doesn't has any arc graphic driver. The machine may not start the desktop correctly, but we can use the ssh to login. 
    # Or you can select 5.19's recovery mode in the grub, then choose resume to resume the normal boot directly.

    sudo apt autoremove

    sudo reboot
    ```

    > <img src="https://llm-assets.readthedocs.io/en/latest/_images/driver_install1.png" width=100%; />

    > <img src="https://llm-assets.readthedocs.io/en/latest/_images/driver_install3.png" width=100%; />

    > <img src="https://llm-assets.readthedocs.io/en/latest/_images/driver_install4.png" width=100%; />

* install drivers

    ```bash
    sudo apt-get update

    sudo apt-get -y install \
        gawk \
        dkms \
        linux-headers-$(uname -r) \
        libc6-dev
        
    # install the latest intel-i915-dkms
    sudo apt-get install -y intel-platform-vsec-dkms intel-platform-cse-dkms intel-i915-dkms intel-fw-gpu

    sudo apt-get install -y gawk libc6-dev udev\
    intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo
    
    sudo reboot
    ```

    > <img src="https://llm-assets.readthedocs.io/en/latest/_images/driver_install5_1.png" width=100%; />

    > <img src="https://llm-assets.readthedocs.io/en/latest/_images/driver_install5_2.png" width=100%; />

* Configuring permissions
    ```bash
    sudo gpasswd -a ${USER} render
    newgrp render

    # Verify the device is working with i915 driver
    sudo apt-get install -y hwinfo
    hwinfo --display
    ```
    > <img src="https://llm-assets.readthedocs.io/en/latest/_images/driver_install6.png" width=100%; />
    
