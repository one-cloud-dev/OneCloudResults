# Официальный CSV Astramis

В карточке отправки №17382 выберите **Скачать** под авторизованной учётной записью с доступом к отправке. Сохраните файл локально как `submissions/submission_volume_wood_v1.csv`. Публичный Score — **78,5013**. [Карточка отправки №17382](https://stackmorelayers.ai/submissions/17382)

Проверка скачанного файла из корня репозитория.

```bash
echo '2975526dadff6a4c0723d1e31d2ef41cb2143cb9f67510feda52243bb1b00950  submissions/submission_volume_wood_v1.csv' | sha256sum -c -
```

Тот же SHA-256 должен получиться при новом инференсе на исходных RGB-D-кадрах. [Инструкция запуска](../README.md#запуск-инференса)
