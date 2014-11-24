*****************************************************************************
Instructions for setting up Visual C++ Express 2008 to compile 64-bit targets
*****************************************************************************

Visual C++ Express 2008 will not build 64-bit targets out of the box. However, this capability can be added by installing the Windows SDK and making some registry edits. Follow the instructions below to do this. Note that if you have the full rather than express version of Visual C++, this should not be necessary.

These instructions have been adapted from the following sources:

http://jenshuebel.wordpress.com/2009/02/12/visual-c-2008-express-edition-and-64-bit-targets/
http://www.cppblog.com/xcpp/archive/2009/09/09/vc2008express_64bit_win7sdk.html
https://github.com/enGits/engrid/wiki/Configure-Microsoft-Visual-Studio-2008-Express-to-also-build-for-Windows-x64
http://wiki.blender.org/index.php/Dev:Doc/Building_Blender/Windows/Visual_C%2B%2B_2008_Express

===============================
Get and install the Windows SDK
===============================

Download the appropriate SDK for your version of Windows. You need version 3.5, later versions will not work with VS2008. The current Windows release can be found here: `Microsoft Windows SDK for Windows 7 and .NET Framework 3.5 SP1 <http://www.microsoft.com/en-us/download/details.aspx?id=3138>`_.

When installing the SDK, you need to select "Windows Headers and Libraries", "Visual C++ Compilers" and "Windows Development Tools". The other items are optional, you may not want to install the documentation as it will take up a lot of space and might take a long time to download due to its large size.

It `appears <http://www.cppblog.com/xcpp/archive/2009/09/09/vc2008express_64bit_win7sdk.html>`_ that on 32-bit Windows, the SDK installer may not install the 64-bit tools the first time around. If so, go to the Control Panel->Programs->Programs and Features, choose to "Change" the "Microsoft Windows SDK for Windows 7 (7.0)" and then click the "Change" option when the installer starts up. Reselect the above options, making sure x64 and IA64 are selected in the sub-categories.

==================================
Edit the registry and rename files
==================================
The Windows SDK installer creates registry keys in ``HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\...`` to point Visual Studio to the components it installed. This is the correct registry location for the full version of Visual Studio, but the express edition uses ``HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VCExpress\...`` instead. Similarly, the SDK installer creates ``*.VCPlatform.config`` files for Visual Studio, but the Express edition expects these to be named ``*.VCPlatform.Express.Config``.

The steps below have been automated by Xia Wei. Get the zip file `here <http://www.cppblog.com/Files/xcpp/VCE64BIT_WIN7SDK.zip>`_, unzip, then open an Administrator command prompt and run ``setup_x86.bat`` or ``setup_x64.bat`` in the unzipped directory. Note that this requires VC++ 2008 to be installed into the default location on the C drive. If you're running on 32-bit Windows or you have another version of the Windows SDK also installed, you may need to use the workarounds `here <http://www.cppblog.com/xcpp/archive/2009/09/09/vc2008express_64bit_win7sdk.html>`_. *You may wish to inspect the contents of these bat files before running code from an unknown source.*

To accomplish the above manually, proceed as follows:

32-bit OS
---------
#. Open the Registry editor (``regedit.exe``).

    #. Visit the key ``HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Visual Studio\9.0\CLSID`` and export it to a file, e.g. ``sdk_data_clsid.reg``.
    #. Visit the key ``HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Visual Studio\9.0\VC`` and export it to a file, e.g. ``sdk_data_vc.reg``.
    #. Visit the key ``HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VCExpress`` and export it to a file as a backup, since the following steps will make changes here.

#. Edit the files ``sdk_data_clsid.reg`` and ``sdk_data_vc.reg``, to replace all occurrences of ``HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Visual Studio`` to ``HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VCExpress``. Save the files.
#. Import the files ``sdk_data_clsid.reg`` and ``sdk_data_vc.reg``, either from Regedit or by double-clicking on the modified files.
#. Go to the folder ``C:\Program Files\Microsoft Visual Studio 9.0\VC\vcpackages`` and rename the file ``AMD64.VCPlatform.config`` to ``AMD64.VCPlatform.Express.config``. Note: if you have installed VC++ 2008 in a different location, you will need to go to that location instead.
    
64-bit OS
---------
#. Open the Registry editor (``regedit.exe``).

    #. Visit the key ``HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Visual Studio\9.0\CLSID`` and export it to a file, e.g. ``sdk_data_clsid.reg``.
    #. Visit the key ``HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Visual Studio\9.0\VC`` and export it to a file, e.g. ``sdk_data_vc.reg``.
    #. Visit the key ``HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\VCExpress`` and export it to a file as a backup, since the following steps will make changes here.

#. Edit the files ``sdk_data_clsid.reg`` and ``sdk_data_vc.reg``, to replace all occurrences of ``HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Visual Studio`` to ``HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\VCExpress``. Save the files.
#. Import the files ``sdk_data_clsid.reg`` and ``sdk_data_vc.reg``, either from Regedit or by double-clicking on the modified files.
#. Go to the folder ``C:\Program Files\Microsoft Visual Studio 9.0\VC\vcpackages`` and rename the file ``AMD64.VCPlatform.config`` to ``AMD64.VCPlatform.Express.config``. Note: if you have installed VC++ 2008 in a different location, you will need to go to that location instead.

=========================================
Make the installed SDK the system default
=========================================
Open the Windows 7 SDK CMD shell located in "Start -> Programs -> Microsoft Windows SDK v7.0 -> CMD Shell" and in the CMD window type the following: 

::
    Setup\WindowsSdkVer.exe -version:v7.0

You may not need this step if you do not have a previous version of the SDK installed.