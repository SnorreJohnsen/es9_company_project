{
  description = "nix shell for es9_company_project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (python3.withPackages (
              ps: with ps; [
                numpy
                matplotlib
                pydantic
                pytest
              ]
            ))
          ];

          shellHook = ''
            			    echo "Entered the es9_company_project development shell"
            				python --version
            			  '';
        };
      }
    );
}
