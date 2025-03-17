namespace Playground2

-- The scoped modifier makes sure the syntax declarations remain in this `namespace`
-- because we will keep modifying this along the chapter
scoped syntax "⊥" : term -- ⊥ for false
scoped syntax "⊤" : term -- ⊤ for true
scoped syntax:40 term " OR " term : term
scoped syntax:50 term " AND " term : term
#check_failure ⊥ OR (⊤ AND ⊥) -- elaboration function hasn't been implemented but parsing passes

end Playground2
