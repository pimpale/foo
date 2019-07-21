(ns hello
  (:gen-class))

(defn Example []
  (println "Hello World")
  (println (+ 1 2))
  nil)

(Example)
