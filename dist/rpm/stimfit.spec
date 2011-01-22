# norootforbuild

Name:           stimfit
Summary:        Stimfit - A program for viewing and analyzing electrophysiological data
Version:        0.10.5
Release:        0
URL:            http://www.stimfit.org
License:        GPL
Group:          Productivity/Scientific/Other
Source:         stimfit-0.10.5.tar.gz

BuildRequires: autoconf automake gcc-c++ blas lapack fftw3-devel
BuildRequires: python-devel python-numpy swig boost-devel
BuildRequires: wxGTK-devel >= 2.8.9
BuildRequires: python-wxGTK >= 2.8.9

Requires: python blas lapack fftw3 libhdf5
Requires: wxGTK >= 2.8.9
Requires: python-wxGTK >= 2.8.9

%description
Stimfit is a free, fast and simple program for viewing and analyzing electrophysiological data. It features an embedded Python shell that allows you to extend the program functionality by using numerical libraries such as NumPy and SciPy.

%prep
%setup -q

%build
export CFLAGS="$RPM_OPT_FLAGS"
export CXXFLAGS="$RPM_OPT_FLAGS"

./configure \
  --enable-python \
  --prefix=%{_prefix}

make %{_smp_mflags}

%install
make install

# Remove static libraries (disable-static seems to leave .la files)
rm -f %{buildroot}/%{_libdir}/%{name}/*.la

# Trash zero length files
for file in $(find %{buildroot} -size 0) ; do
    rm -f "$file"
done

# Remove empty directories
find %{buildroot}/%{_datadir}/%{name} -type d -empty -delete

%post
ldconfig

%postun
ldconfig

%clean
%__rm -rf %buildroot

%files
%defattr(-,root,root,-)
%{_bindir}/*
%{_libdir}/*

%changelog