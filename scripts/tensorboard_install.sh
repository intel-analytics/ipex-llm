#!/bin/bash

iprt='\e[32m[installation]\e[37m'
eprt='\e[31m[ERROR!]\e[37m'
wprt='\e[33m[WARNNING!]\e[37m'
iecho () { printf "$iprt $1\n"; }
eecho () { printf "$eprt $1\n"; }
wecho () { printf "$wprt $1\n"; }

help_str="Usage: $0 [-p your/python/binary/path]\n            Example: $0 -p \$(which python)"
if [ $# -eq 0 ]; then
	wecho "$help_str"
	exit 1
fi

if [ ${1:0:1} != "-" ]; then 
	wecho "$help_str"
	exit 1
fi

while getopts "p:" option;
do
	case "$option" in
		p)	
			pythonPath="$OPTARG" ;;
		[?])
			wecho "$help_str"
			exit 1 ;;
	esac
done

iecho "Checking whether Tensorboard is installed"
if [ -n "$(which tensorboard)" ]; then
	tbd_pbp=$(head -n 1 "$(which tensorboard)")
	ori_pbp=$pythonPath
	iecho "tensorboard is launched by ${tbd_pbp#\#!}"
	iecho "current python binary path is $ori_pbp"
	if [ "${tbd_pbp#\#!}" == "$ori_pbp" ]; then
		iecho "Tensorboard is already installed!"
		iecho "To keep installation, please remove current tensorboard"
		wecho "Exit without installation" >&2
		exit 1
	else
		wecho "Tensorboard is installed with ${tbd_pbp#\#!} instead of ${ori_pbp}"
		iecho "Will install tensorboard with $ori_pbp"
	fi
else
	iecho "Tensorboard is not installed. Keep installing"
fi

iecho "Checking requirement: pip"
$pythonPath -m pip 1>/dev/null 2>/dev/null; flag=$?;
if [ $flag != 0 ]; then
	wecho "pip is not installed for $pythonPath"
	iecho "Trying to install pip"
	iecho "Installing pip"
:<< !
	(set -x; sudo -E apt install python3-pip -y 2>/tmp/installpython3pip; echo $? > /tmp/exitstate)
	flag=$(grep 'Unable to locate package' /tmp/installpython3pip);
	exit_state=$(cat /tmp/exitstate);
	if [ -n "$flag" -o ! exit_state == 0 ]; then
		if [ -n "$flag" ]; then
			wecho "Unable to locate package python3-pip. Maybe the apt repository list is outdated?"
		fi
		if [ ! exit_state == 0 ]; then
			wecho "Unable to install python3-pip. Maybe the apt repository list is outdated?"
		fi
		iecho "Tring to update apt repository list"
		iecho "Updating apt repository list ..."
		(set -x;sudo -E apt update)
		iecho "Trying to install pip3 again"
		iecho "Installing python3-pip"
		(set -x;sudo -E apt-get install python3-pip -y)
	fi
!
	[ ! -f /tmp/getpippy ] && wget https://bootstrap.pypa.io/get-pip.py -O /tmp/getpippy
	$pythonPath /tmp/getpippy
	iecho "Checking requirement: pip"
	$pythonPath -m pip 1>/dev/null 2>/dev/null; flag=$?;
	if [ $flag != 0 ]; then
		eecho "ERROR! After trying installation, pip is still not found. Please install it first" >&2
		exit 1
	fi
fi
iecho "pip found"

version=$($pythonPath -m pip -V | awk '{print $2}')
iecho "Checking current pip version"
iecho "current pip version is $version"
if [ $version \< "9.0.1" ]; then
	wecho "The pip-$version cannot support the installation."
	iecho "Trying to upgrade pip-$version to the latest version"
	(set -x; export LC_ALL=C; sudo -E $pythonPath -m pip install --upgrade pip; unset LC_ALL)
	version=$(pip -V | awk '{print $2}')
	iecho "Checking current pip version"
	iecho "current pip version is $version"
	if [ $version \< "9.0.1" ]; then
		eecho "The pip-$version still cannot support the installation. Please update pip to the latest version [>= 9.0.1]" >&2
		exit 1
	fi
fi
iecho "pip latest"

iecho "Installing tensorboard"
if [ -f "$(which tensorboard)" ]; then
	cp $(which tensorboard) $(which tensorboard)"_backup"
fi
$pythonPath -m pip install tensorboard==1.0.0a5
if [ $? != 0 ]; then
	eecho "tensorboard installation failed. Please try to fix the error"
	mv $(which tensorboard)"_backup" $(which tensorbaord) 2>/dev/null
	exit 1
else
	iecho "tensorboard intsalled successfully"
fi

iecho "Modifying the entry to tensorboard"
(set -x; sed -i "/^PYTHON_BINARY/c PYTHON_BINARY = \'$pythonPath\'" $(which tensorboard))
iecho "Test running tensorboard"
(set -x; timeout 5 tensorboard --logdir=/tmp/tensorboard)
if [ $? == 124 ]; then
	iecho "Running successfully!"
	iecho "Tensorboard is correctly installed"
elif [ $? != 0 ]; then
	eecho "Running test ended with failure"
	exit 1
fi
