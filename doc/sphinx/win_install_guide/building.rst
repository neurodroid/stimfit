****************
Building Stimfit
****************

:Author: Christoph Schmidt-Hieber
:Date:  |today|

===================
Get the source code
===================

Clone the latest source code into your home directory (on cygwin, this will be something like /cygdrive/c/Users/username/)

::

    $ cd /cygdrive/c/Users/username
    $ git clone https://code.google.com/p/stimfit/

=============
Build Stimfit
=============

Open the solution in stimfit/windows/VS2008/Stimfit/Stimfit.sln with Visual C++ Express 2008. Build the solution by clicking "Build" -> "Build Solution". Alternatively, open the Visual Studio Command Prompt:

::

    C:\> cd C:\Users\username\stimfit\windows\VS2008\Stimfit
    C:\Users\username\stimfit\windows\VS2008\Stimfit> msbuild Stimfit.sln /p:Configuration=Release

===================
Create an installer
===================

Use nsis to compile the installer script in stimfit/nsis/installer.nsi.
