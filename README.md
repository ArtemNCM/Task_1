# Nebula AI classifier

MVP класифікує звернення клієнтів за категорією, пріоритетом і рекомендованим маршрутом. Ризикові та неоднозначні випадки передаються на перевірку людині. Refund, видалення даних і відповіді клієнтам автоматично не виконуються.

Підсумкова таблиця прогонів п’яти моделей: [порівняння](docs/comparison.md).

## Запуск

Потрібен Python 3.11+. У PowerShell з кореня проєкту:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Додайте `OPENROUTER_API_KEY` у `.env`, потім класифікуйте звернення:

```powershell
python classify.py --text "Please cancel my subscription" --allow-live
```

Офлайн-оцінювання використовує mock-модель і не потребує API-ключа:

```powershell
python evaluate.py
```

Для live-оцінювання потрібні ключ і дозвіл на платні запити: `python evaluate.py --model gpt-6-luna --allow-live`. Результати зберігаються локально в `01/results` або `02/results` і не додаються до Git.

Порівняйте live-прогони, передавши їхні каталоги з результатами. За замовчуванням виводиться лише коротка Markdown-таблиця в термінал:

```powershell
python compare.py 01/results/run-1 01/results/run-2
```

Можна порівнювати прогони з `02/results` тією самою командою. Для класифікації без тексту в команді використайте `--file шлях\до\файлу.txt`.
