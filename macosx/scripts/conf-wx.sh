arch_flags="-arch x86_64 -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5"

../configure CC=gcc-4.2 CXX=g++-4.2 LD=g++-4.2 CPPFLAGS="" CFLAGS="$arch_flags" CXXFLAGS="$arch_flags" LDFLAGS="$arch_flags" OBJCFLAGS="$arch_flags" OBJCXXFLAGS="$arch_flags" --with-macosx-version-min=10.5 --prefix=/Users/cs/wxbin --with-osx_cocoa --with-opengl --enable-geometry --enable-graphics_ctx --enable-sound --with-sdl --enable-mediactrl --enable-std_string 
