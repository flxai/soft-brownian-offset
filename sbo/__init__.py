from pbr.version import VersionInfo

__version__ = VersionInfo('sbo').version_string()
__version_info__ = VersionInfo('sbo').semantic_version().version_tuple()
