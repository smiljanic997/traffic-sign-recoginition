import subprocess
import re

def is_device_present(device_id):
    device_re = re.compile("Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)

    lsusb = subprocess.check_output('lsusb')
    devices = []

    for i in lsusb.decode().split('\n'):
        if i:
            info = device_re.match(i)
            if info:
                dinfo = info.groupdict()
                dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
                devices.append(dinfo)

    device_ids = [devices[i]['id'] for i in range(len(devices))]
    return device_id in device_ids