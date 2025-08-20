#!/bin/bash

CONFIGFS="/sys/kernel/config"
GADGET="$CONFIGFS/usb_gadget"
VID="0x05c6"
PID="0x90cb" 

# find serialno from cmdline
SERIAL=$(grep -o "androidboot.serialno=[A-Za-z0-9]*" /proc/cmdline | cut -d "=" -f2)
if [ -z "$SERIAL" ]; then
    echo "SERIAL not found with cmdline, use default"
    SERIAL="12345678"
fi

MANUF="Luxonis"
PRODUCT="Luxonis UVC Camera"
UDC=$(ls /sys/class/udc) # will identify the 'first' UDC

echo "Detecting platform:"
echo "  product : $PRODUCT"
echo "  udc     : $UDC"
echo "  serial  : $SERIAL"

remove_all_gadgets() {
    echo Removing all gadget configs

    if [ ! -d /sys/kernel/config/usb_gadget/g1 ]; then
        echo "No gadget found, nothing to remove"
        return 0
    fi

    pushd /sys/kernel/config/usb_gadget/g1 >/dev/null

    echo "Unbinding USB Device Controller..."
    retries=0
    max_retries=5
    while true; do
        echo "" > UDC 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "Successfully unbound UDC"
            break
        fi
        retries=$((retries + 1))
        if [ $retries -ge $max_retries ]; then
            echo "Failed to unbind UDC after $max_retries attempts"
            break
        fi
        sleep 1
    done

    rm configs/c.1/* 2>/dev/null
    rmdir configs/c.1/strings/0x409 2>/dev/null
    rmdir configs/c.1 2>/dev/null

    rm functions/uvc.0/streaming/header/h/* 2>/dev/null
    rm functions/uvc.0/streaming/header/h1/* 2>/dev/null
    rm functions/uvc.0/streaming/header/h/m 2>/dev/null
    rmdir functions/uvc.0/streaming/h264/h/* 2>/dev/null
    rmdir functions/uvc.0/streaming/h264/h 2>/dev/null
    rmdir functions/uvc.0/streaming/mjpeg/m/* 2>/dev/null
    rmdir functions/uvc.0/streaming/mjpeg/m1/* 2>/dev/null
    rmdir functions/uvc.0/streaming/mjpeg/m 2>/dev/null
    rmdir functions/uvc.0/streaming/mjpeg/m1 2>/dev/null
    rmdir functions/uvc.0/streaming/uncompressed/u/* 2>/dev/null
    rmdir functions/uvc.0/streaming/uncompressed/u1/* 2>/dev/null
    rmdir functions/uvc.0/streaming/uncompressed/u 2>/dev/null
    rmdir functions/uvc.0/streaming/uncompressed/u1 2>/dev/null

    rm functions/uvc.0/streaming/class/fs/* 2>/dev/null
    rm functions/uvc.0/streaming/class/hs/* 2>/dev/null
    rm functions/uvc.0/streaming/class/ss/* 2>/dev/null

    rmdir functions/uvc.0/streaming/header/h 2>/dev/null
    rmdir functions/uvc.0/streaming/header/h1 2>/dev/null

    rm functions/uvc.0/control/class/fs/* 2>/dev/null
    rm functions/uvc.0/control/class/ss/* 2>/dev/null
    rmdir functions/uvc.0/control/header/* 2>/dev/null

    rmdir functions/* 2>/dev/null

    rmdir strings/0x409

    popd >/dev/null

    rmdir /sys/kernel/config/usb_gadget/g1

    if [ ! -e /sys/kernel/config/usb_gadget/g1 ]; then
        echo "Gadget successfully removed!"
    else
        echo "Failed to remove gadget!"
        exit 1
    fi
}

create_frame() {
    # Example usage:
    # create_frame <function name> <width> <height> <format> <name>

    FUNCTION=$1
    WIDTH=$2
    HEIGHT=$3
    FORMAT=$4
    NAME=$5

    wdir=functions/$FUNCTION/streaming/$FORMAT/$NAME/${HEIGHT}p

    mkdir -p "$wdir"
    echo "$WIDTH" > "$wdir/wWidth"
    echo "$HEIGHT" > "$wdir/wHeight"
    echo $(( WIDTH * HEIGHT * 2 )) > "$wdir/dwMaxVideoFrameBufferSize"
    cat <<EOF > "$wdir/dwFrameInterval"
333333
EOF
}

create_uvc() {
    # Example usage:
    #   create_uvc <target config> <function name>
    #   create_uvc config/c.1 uvc.0
    CONFIG=$1
    FUNCTION=$2

    echo "    Creating UVC gadget functionality : $FUNCTION"
    mkdir "functions/$FUNCTION"

    # create_frame "$FUNCTION" 640 360 uncompressed u
    # create_frame "$FUNCTION" 1280 720 uncompressed u
    # create_frame "$FUNCTION" 320 180 uncompressed u
    create_frame "$FUNCTION" 1920 1080 mjpeg m
    # create_frame "$FUNCTION" 640 480 mjpeg m
    # create_frame "$FUNCTION" 640 360 mjpeg m

    mkdir "functions/$FUNCTION/streaming/header/h"
    cd "functions/$FUNCTION/streaming/header/h"
    # ln -s ../../uncompressed/u
    ln -s ../../mjpeg/m
    cd ../../class/fs
    ln -s ../../header/h
    cd ../../class/hs
    ln -s ../../header/h
    cd ../../class/ss
    ln -s ../../header/h
    cd ../../../control
    mkdir header/h
    ln -s header/h class/fs
    ln -s header/h class/ss
    cd ../../../

    echo 3072 > "functions/$FUNCTION/streaming_maxpacket"

    ln -s "functions/$FUNCTION" configs/c.1
}

terminate() {
    if [ "$child_pid" -gt 0 ] 2>/dev/null && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid"
    wait "$child_pid" || true
    fi
    exit 143
}
trap terminate INT TERM

case "$1" in
    start)
    echo "    ==== Configuring USB gadget ===="
    sleep 5

    if [ ! -d "$CONFIGFS" ]; then
        echo "Configfs not mounted, please mount it first"
        exit 1
    fi

    remove_all_gadgets

    echo "Creating the USB gadget"

    echo "Creating gadget directory g1"
    mkdir -p "$GADGET/g1"

    pushd "$GADGET/g1" >/dev/null
    if [ $? -ne 0 ]; then
        echo "Error creating usb gadget in configfs"
        exit 1
    else
        echo "OK"
    fi

    echo "Setting Vendor and Product ID's"
    echo "$VID" > idVendor
    echo "$PID" > idProduct
    echo "OK"

    echo "Setting English strings"
    mkdir -p strings/0x409
    echo "$SERIAL" > strings/0x409/serialnumber
    echo "$MANUF"  > strings/0x409/manufacturer
    echo "$PRODUCT" > strings/0x409/product
    echo "OK"

    echo "Setting max speed to super-speed (5 Gbps)"
    echo "super-speed" > max_speed

    echo "Creating Config"
    mkdir configs/c.1
    mkdir configs/c.1/strings/0x409

    echo "Creating functions..."
    create_uvc configs/c.1 uvc.0
    echo "OK"

    echo "Binding USB Device Controller..."
    while true; do
        echo "$UDC" > UDC 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "Successfully bound UDC controller $UDC"
            break
        fi
        sleep 1
    done

    popd >/dev/null

    echo "    ==== Configuration done ===="
    
    echo "    ==== Starting UVC APP ===="
    retries=0
    max_retries=5
    backoff=5
    child_pid=0

    while [ $retries -lt $max_retries ]; do
      /app/uvc_example &
      child_pid=$!
      wait "$child_pid"
      status=$?

      if [ $status -eq 0 ]; then
        echo "uvc_example exited normally."
        exit 0
      fi

      retries=$((retries + 1))
      printf '[%(%Y-%m-%d %H:%M:%S)T] uvc_example exited with status %d. Restarting (%d/%d) in %ds...\n' -1 "$status" "$retries" "$max_retries" "$backoff"
      sleep "$backoff"
    done

    echo "uvc_example failed $max_retries times. Not restarting anymore."
    exit 1
    ;;

    stop)
    echo "    ==== Stopping the USB gadget ===="
    remove_all_gadgets
    ;;
    *)
    echo "Usage : $0 {start|stop}"
    ;;
esac
