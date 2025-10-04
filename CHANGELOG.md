# Changelog

All notable changes to this project will be documented in this file.

## [1.0.10] - 2025-10-04
### Added
- Initial Docker support (single `Dockerfile`) with CLI entrypoint `uls_predict_image`.
- GitHub Actions workflow (`docker.yml`) to build and publish image to GitHub Container Registry (tags: `latest`, version, commit SHA).
- This `CHANGELOG.md` file.

### Changed
- Simplified container build approach.

## [1.0.4] - 2025-03-14
### Added
- Initial public release of the `unet_lungs_segmentation` package with CLI tools `uls_predict_image` and `uls_predict_folder`.

---
