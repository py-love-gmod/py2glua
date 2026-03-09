# API YAML Schema (Prototype)

Этот файл описывает текущий формат `data/api/**/*.yml`, из которого собирается `build_api`.

## 1. Корневые ключи

```yml
module: api/gmod_api.py
imports:
  - from typing import TypeVar
type_vars:
  - name: F
    bound: Callable[..., object]
functions: []
classes: []
```

- `module` (обязательно): путь выходного python-модуля внутри `build_api`.
- `imports`: список строк `import/from import`.
- `type_vars`: объявления module-level переменных типа (`F = TypeVar(...)`).
- `functions`: функции верхнего уровня.
- `classes`: классы верхнего уровня.

Если в `type_vars` используется `TypeVar(...)`, импорт `from typing import TypeVar` добавится автоматически (если его нет в `imports`).

## 2. `type_vars` (упрощенный TypeVar)

Поддерживаются 2 режима:

1. Через `expr` (полный контроль):

```yml
type_vars:
  - name: P
    expr: ParamSpec("P")
```

2. Через поля (упрощенно):

```yml
type_vars:
  - name: F
    bound: Callable[..., object]

  - name: T
    constraints: [int, str]
    covariant: true
```

Доступные ключи:

- `name` (обязательно)
- `expr` (альтернатива всему остальному)
- `factory` (по умолчанию `TypeVar`)
- `args` (`list`)
- `kwargs` (`mapping`)
- `bound` (`str`)
- `constraints` (`list[str]`)
- `covariant` (`bool`)
- `contravariant` (`bool`)

## 3. Функции

```yml
functions:
  - name: include
    args:
      - name: path
        type: str
    returns: nil_type
    body: pass
    decorators:
      - kind: gmod_api
        name: include
        realms: [SHARED]
        method: false
```

### `body`

- `pass`
- `ellipsis`
- `decorator` (шаблон `def decorator(fn): return fn; return decorator`)
- `raw` (с `body_lines`)

```yml
body: raw
body_lines:
  - return nil
```

## 4. Классы

```yml
classes:
  - name: SWEP
    decorators:
      - kind: raw
        expr: CompilerDirective.gmod_prototype("weapons")
    bases: [BaseClass]
    fields: []
    methods: []
    classes: []
    ifs: []
```

Поддерживается:

- вложенность классов через `classes`
- декораторы класса через `decorators`
- наследование через `bases`
- условные блоки внутри класса через `ifs`

### `ifs` внутри класса

```yml
ifs:
  - condition: Realm.CLIENT
    fields:
      - name: Category
        type: str
        value: "PLG SWEP"
    methods:
      - name: ViewModelDrawn
        args:
          - name: self
            type: SWEP
        body: pass
    classes: []
```

## 5. Локализация

Локализация хранится отдельно от API:

- `data/loc_meta.yml`
- `data/loc/<lang>/**/*.yml`

Поддерживаются секции:

- `functions`
- `classes`
- `methods`
- `fields`

Докстринги собираются из локализации автоматически.
