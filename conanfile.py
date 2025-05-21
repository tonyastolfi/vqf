from conan import ConanFile
import os, sys, platform


class VqfRecipe(ConanFile):
    name = "vqf"

    python_requires = "cor_recipe_utils/0.8.7"
    python_requires_extend = "cor_recipe_utils.ConanFileBase"

    tool_requires = [
        "cmake/[>=3.20.0]",
    ]

    settings = "os", "compiler", "build_type", "arch"

    exports_sources = [
        "CMakeLists.txt",
        "**/CMakeLists.txt",
        "src/*.hpp",
        "src/**/*.hpp",
        "src/*.ipp",
        "src/**/*.ipp",
        "src/*.cpp",
        "src/**/*.cpp",
        "src/*.h",
        "src/**/*.h",
        "src/*.c",
        "src/**/*.c",
    ]

    #+++++++++++-+-+--+----- --- -- -  -  -   -
    # Optional metadata
    #
    license = "BSD-3"

    author = "Prashant Pandey, Alex Conway, Rob Johnson, and Tony Astolfi"

    url = "https://github.com/tonyastolfi/vqf"

    description = "Vector Quotient Filters: Overcoming the Time/Space Trade-Off in Filter Design (C++ fork)"

    topics = ("data structures",)
    #
    #+++++++++++-+-+--+----- --- -- -  -  -   -

    def requirements(self):
        VISIBLE = self.cor.VISIBLE
        OVERRIDE = self.cor.OVERRIDE

        self.requires("openssl/[>=3.2.0 <4]", **VISIBLE)
        
        self.test_requires("batteries/[>=0.59.0 <1]")
        self.test_requires("boost/[>=1.84.0 <2]")
        self.test_requires("gtest/[>=1.14.0 <2]")

    def configure(self):
        self.options["gtest"].shared = False
        self.options["boost"].shared = False
        self.options["boost"].without_test = True
        self.options["batteries"].with_glog = True
        self.options["batteries"].header_only = False

    #+++++++++++-+-+--+----- --- -- -  -  -   -

    def set_version(self):
        return self.cor.set_version_from_git_tags(self)

    def layout(self):
        return self.cor.layout_cmake_unified_src(self)

    def generate(self):
        return self.cor.generate_cmake_default(self)

    def build(self):
        return self.cor.build_cmake_default(self)

    def package(self):
        return self.cor.package_cmake_lib_default(self)

    def package_info(self):
        return self.cor.package_info_lib_default(self)

    def package_id(self):
        return self.cor.package_id_lib_default(self)

    #+++++++++++-+-+--+----- --- -- -  -  -   -
