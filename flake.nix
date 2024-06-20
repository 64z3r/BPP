{
  description = "An FHS shell with PDM and all necessary runtime dependencies";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, home-manager }:
    let
      # pkgs = nixpkgs.legacyPackages.x86_64-linux;
      pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
      name = "Python 3.11 FHS";
      venv = ".venv";
      project = "PA";
      # nvidia_x11 = pkgs.linuxKernel.packages.linux_zen.nvidia_x11;
      nvidia_x11 = pkgs.linuxPackages.nvidia_x11;
    in {
      devShell.x86_64-linux = (pkgs.buildFHSUserEnv {
        name = name;
        targetPkgs = pkgs: (
          with pkgs; [
            autoconf
            binutils
            boost
            boost.dev
            conda
            coreutils
            cudatoolkit
            curl
            direnv
            dssp
            freeglut
            freeglut.dev
            freetype
            freetype.dev
            gcc
            git
            gitRepo
            glew
            glew.dev
            glm
            gnumake
            gnupg
            gperf
            graphviz
            libGL
            libGL.dev
            libGLU
            libGLU.dev
            libpng
            libpng.dev
            libselinux
            libxml2
            libxml2.dev
            m4
            mmtf-cpp
            msgpack-c
            msgpack-cxx
            ncurses5
            netcdf
            nvidia_x11
            openbabel2
            openblas
            pdm
            procps
            pymol
            stdenv.cc
            unzip
            util-linux
            wget
            xorg.libICE
            xorg.libSM
            xorg.libX11
            xorg.libXext
            xorg.libXi
            xorg.libXmu
            xorg.libXrandr
            xorg.libXrender
            xorg.libXv
            zlib
            (pkgs.python311.withPackages (
              ps: with ps; [
                ipython
                pip
                setuptools
                virtualenv
              ]
            ))
            (stdenv.mkDerivation {
              name = "msms";
              src = fetchurl {
                url = "https://ccsb.scripps.edu/msms/download/933/";
                sha256 = "sha256-XwylA2C1k450xTjgOZ1YKrxKQO9M9BDmbzGh+R5uPh8=";
              };
              nativeBuildInput = [ autoPatchelfHook ];
              phases = [ "installPhase" ];
              installPhase = ''
                tar xzf $src
                install -D ./msms.x86_64Linux2.2.6.1 $out/bin/msms
                install -D ./msms.1 $out/share/man/man1/msms.1
                chmod a+x $out/bin/msms
              '';
            })
          ]
        );
        profile = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          export EXTRA_LDFLAGS="-L/lib -L${nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
          export FONTCONFIG_FILE=/etc/fonts/fonts.conf
          export PYTHONPATH=$PYTHONPATH:/usr/lib/python
          export QTCOMPOSE=${pkgs.xorg.libX11}/share/X11/locale
        '';
        runScript = ''fish -C "
          function fish_right_prompt
            set_color green
            echo '${name}'
            set_color normal
          end
          
          python -m venv "${venv}" --prompt="${project}"
          source "${venv}/bin/activate.fish"
          
          python -m pip install --upgrade pip
          python -m pip install ipykernel
          python -m ipykernel install --user --name="${project}"
        "'';
      }).env;
    };
}
