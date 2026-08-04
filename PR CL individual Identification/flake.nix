{
  description = "Pattern Recognition - Individual Identification";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # Override jaraco-test to skip its tests (known issue with pytest 8.4.2)
        python312Packages = pkgs.python312Packages.override {
          overrides = self: super: {
            jaraco-test = super.jaraco-test.overridePythonAttrs (old: {
              doCheck = false;
            });
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            python312Packages.pip
            pkgs.pyright

            python312Packages.numpy
            python312Packages.pandas
            # python312Packages.geopandas - removed, installed via pip
            python312Packages.shapely
            python312Packages.kagglehub
            python312Packages.python-dotenv
            python312Packages.wandb 
            python312Packages.torchWithRocm
            python312Packages.h5py
            python312Packages.torchvision 
            python312Packages.timm
            python312Packages.matplotlib

            python312Packages.pytest
            
            # GUI support for matplotlib (optional, for plt.show())
            python312Packages.tkinter
            
            # System libraries for OpenCV (cv2)
            pkgs.libGL
            pkgs.libGLU
            pkgs.xorg.libX11
            pkgs.xorg.libXext
            pkgs.xorg.libXrender
            pkgs.glib
            pkgs.gst_all_1.gstreamer
            pkgs.gst_all_1.gst-plugins-base
          ];

          shellHook = ''
            # Set up library paths for OpenCV
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.libGL
              pkgs.libGLU
              pkgs.xorg.libX11
              pkgs.xorg.libXext
              pkgs.xorg.libXrender
              pkgs.glib
              pkgs.gst_all_1.gstreamer
              pkgs.gst_all_1.gst-plugins-base
            ]}:$LD_LIBRARY_PATH
            
            # Only set up venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Setting up Python environment..."
              python3 -m venv .venv
            fi

            echo "Activating Python environment..."
            source .venv/bin/activate

            # Only install if requirements changed or venv is new
            if [ ! -f .venv/.requirements-installed ] || [ requirements.txt -nt .venv/.requirements-installed ]; then
              echo "Installing dependencies..."
              pip install -r requirements.txt
              touch .venv/.requirements-installed
            fi
          '';
        };
      }
    );
}