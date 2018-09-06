#!/bin/bash

grep -rli $1 * | xargs -i@ sed -i 's/$1/$2/g' @
