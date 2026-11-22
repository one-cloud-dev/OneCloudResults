# Веса финального решения

Скопируйте файлы в этот каталог под именами из таблицы. Пути в рабочей области указаны относительно `/home/jovyan/work`. Откройте **Files** в рабочей области OneCloud на StackMoreLayers, если команда предоставила вам доступ. Для внешнего запуска скачайте открытые веса DINOv2 из источника ниже. Если у вас нет доступа к рабочей области, запросите командные веса у OneCloud через организаторов.

| Имя в `models/` | Источник |
| --- | --- |
| `best.pt` | `onecloud_v2_20261120T220121/verified_candidate_v1/best.pt` |
| `material.joblib` | `onecloud_v2_20261120T220121/verified_candidate_v1/material.joblib` |
| `head_all40.npz` | `runs/dinov2_material_stage1/head_all40.npz` |
| `head_train32.npz` | `runs/dinov2_material_stage1/head_train32.npz` |
| `volume_train32.cbm` | `runs/rgbd_volume_stage2/volume_train32.cbm` |
| `take_train32_strict.joblib` | `runs/take_stage3/take_train32_strict.joblib` |
| `dinov2_vits14_pretrain.pth` | [Чекпойнт DINOv2 ViT-S/14 от Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |

Проверенные SHA-256 всех семи файлов записаны в [`artifacts.json`](artifacts.json). Скрипт сверяет их перед загрузкой. Для моделей `.pt` и `.joblib` используйте только проверенные файлы команды.

Скачайте [исходники DINOv2 версии `7764ea0f912e53c92e82eb78a2a1631e92725fc8`](https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.zip). SHA-256 архива `04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48`. Выполните проверку и распаковку из корня репозитория.

```bash
mkdir -p vendor
curl -fL https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.zip \
  -o vendor/dinov2-source.zip
echo '04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48  vendor/dinov2-source.zip' | sha256sum -c -
unzip vendor/dinov2-source.zip -d vendor
```

В авторизованной среде исходники уже распакованы в `runs/material_dinov2_v1_20261122/vendor/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Этот путь можно передать в `--dino-source`. [Манифест оригинального запуска](../experiments/astramis/dinov2_material_stage1/encoder.json)
