# Changelog

All notable changes to Timbrel are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.0.2]

### Added

- Speaker encoder (d-vector) and prosody encoder with FiLM output.
- Disentanglement toolkit: gradient reversal, adversarial speaker classifier,
  CLUB mutual-information estimator.
- Variance adaptor (duration/pitch/energy) and speaker-conditional mel decoder.
- Aggregate training loss and a minimal `Trainer`.

## [0.0.1]

### Added

- Project scaffolding, CI, and ruff/mypy/pytest tooling.
- Bilingual (Mandarin + English) text front-end and unified phoneme inventory.
- Log-mel / f0 / energy feature extraction (numpy + torch, no librosa).
